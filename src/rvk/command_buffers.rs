use std::{
    fmt::Display,
    sync::{Arc, Mutex},
};

use ash::vk::{self, Handle};
use thiserror::Error;

use crate::rvk::device::Device;
use crate::LockResultExt;
// no direct sync object imports here; submission moved to Device

/// Level of command buffer to allocate (maps to Vulkan's CommandBufferLevel)
#[derive(Clone, Copy, Debug)]
pub enum CommandBufferLevel {
    Primary,
    Secondary,
}

impl From<CommandBufferLevel> for vk::CommandBufferLevel {
    fn from(l: CommandBufferLevel) -> vk::CommandBufferLevel {
        match l {
            CommandBufferLevel::Primary => vk::CommandBufferLevel::PRIMARY,
            CommandBufferLevel::Secondary => vk::CommandBufferLevel::SECONDARY,
        }
    }
}

#[derive(Debug)]
pub struct CommandPool {
    parent_device: Arc<Device>,
    pool: Mutex<vk::CommandPool>,
}

/// Error returned when allocating and beginning a command buffer in one step.
#[derive(thiserror::Error, Debug)]
pub enum BeginCommandBufferError {
    #[error("Allocation failed: {0}")]
    Allocate(#[from] AllocateCommandBuffersError),
    #[error("Vulkan error while beginning command buffer: {0}")]
    Begin(ash::vk::Result),
}

#[derive(Error, Debug)]
pub enum CommandPoolCreateError {
    #[error("Unknown Vulkan Error {0}")]
    UnknownVulkan(vk::Result),
}

impl CommandPool {
    pub fn new(
        parent_device: &Arc<Device>,
        family_index: u32,
    ) -> Result<Self, CommandPoolCreateError> {
        use CommandPoolCreateError as Error;
        let ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        //SAFETY: Valid CI
        let cp = unsafe { parent_device.inner().create_command_pool(&ci, None) }
            .map_err(Error::UnknownVulkan)?;

        Ok(Self {
            parent_device: parent_device.clone(),
            pool: Mutex::new(cp),
        })
    }
    pub fn allocate_command_buffers(
        self: &Arc<Self>,
        count: u32,
        level: CommandBufferLevel,
    ) -> Result<Vec<CommandBuffer>, AllocateCommandBuffersError> {
        use AllocateCommandBuffersError as Error;
        use AllocateCommandBuffersErrorType as ErrorType;
        // Get exclusive access to the pool
        let pool = self.pool.lock().unwrap_or_else(|e| {
            tracing::warn!("command pool mutex poisoned, recovering: {:#?}", e);
            e.into_inner()
        });
        let pool = *pool;
        let cbai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(count)
            .level(level.into());

        //SAFETY: pool comes from parent device, have exclusive access,
        //valid allocate_info
        let cbs =
            unsafe { self.parent_device.inner().allocate_command_buffers(&cbai) }.map_err(|e| {
                Error {
                    e: match e {
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => ErrorType::MemoryExhaustion,
                        _ => ErrorType::UnknownVk(e),
                    },
                    count,
                    pool,
                }
            })?;

        Ok(cbs
            .iter()
            .map(|raw_cb| CommandBuffer {
                parent_pool: self.clone(),
                inner: *raw_cb,
            })
            .collect())
    }

    /// Allocate a single command buffer from this pool and return a RAII wrapper.
    pub fn allocate_command_buffer(
        self: &Arc<Self>,
        level: CommandBufferLevel,
    ) -> Result<CommandBuffer, AllocateCommandBuffersError> {
        let mut v = self.allocate_command_buffers(1, level)?;
        Ok(v.remove(0))
    }

    /// Convenience: allocate `count` primary command buffers.
    pub fn allocate_command_buffers_primary(
        self: &Arc<Self>,
        count: u32,
    ) -> Result<Vec<CommandBuffer>, AllocateCommandBuffersError> {
        self.allocate_command_buffers(count, CommandBufferLevel::Primary)
    }

    /// Convenience: allocate a single primary command buffer.
    pub fn allocate_command_buffer_primary(
        self: &Arc<Self>,
    ) -> Result<CommandBuffer, AllocateCommandBuffersError> {
        self.allocate_command_buffer(CommandBufferLevel::Primary)
    }

    /// Allocate a primary command buffer and begin recording with
    /// ONE_TIME_SUBMIT usage. Returns a `RecordingCommandBuffer` on success.
    pub fn allocate_and_begin_command_buffer_primary(
        self: &Arc<Self>,
    ) -> Result<RecordingCommandBuffer, BeginCommandBufferError> {
        let cb = self.allocate_command_buffer_primary()?;
        // Map vk::Result into BeginCommandBufferError via map_err
        cb.begin_one_time().map_err(BeginCommandBufferError::Begin)
    }
}
impl Drop for CommandPool {
    fn drop(&mut self) {
        //SAFETY: Exclusive access to pool, pool was created from parent_device
        unsafe {
            self.parent_device.inner().destroy_command_pool(
                *self.pool.get_mut().unwrap_or_else(|e| {
                    tracing::warn!(
                        "command pool get_mut poisoned during drop, recovering: {:#?}",
                        e
                    );
                    e.into_inner()
                }),
                None,
            )
        };
    }
}

#[derive(Debug, Error)]
pub struct AllocateCommandBuffersError {
    count: u32,
    pool: vk::CommandPool,
    e: AllocateCommandBuffersErrorType,
}

impl Display for AllocateCommandBuffersError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Error while allocating {} command buffers from pool {}: {}",
            self.count,
            self.pool.as_raw(),
            self.e
        )
    }
}

#[derive(Debug, Error)]
pub enum AllocateCommandBuffersErrorType {
    #[error("Could not allocate due to memory exhaustion")]
    MemoryExhaustion,
    #[error("Unknown vulkan error {0}")]
    UnknownVk(vk::Result),
}

#[derive(Debug)]
pub struct CommandBuffer {
    parent_pool: Arc<CommandPool>,
    inner: vk::CommandBuffer,
}

impl CommandBuffer {
    /// Get the raw `vk::CommandBuffer` handle.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.inner
    }

    /// Begin recording on this command buffer. Consumes the `CommandBuffer`
    /// and returns a `RecordingCommandBuffer` on which recording operations
    /// may be performed. `usage` should be a combination of
    /// `vk::CommandBufferUsageFlags` (e.g. ONE_TIME_SUBMIT).
    pub fn begin(
        self,
        usage: vk::CommandBufferUsageFlags,
    ) -> Result<RecordingCommandBuffer, vk::Result> {
        let device = self.parent_pool.parent_device.inner();
        let begin_info = vk::CommandBufferBeginInfo::default().flags(usage);

        // SAFETY: we have a valid command buffer allocated on this device
        unsafe { device.begin_command_buffer(self.inner, &begin_info) }?;
        let cb = self.inner;
        let parent_pool = self.parent_pool.clone();
        //Prevent us from dropping ourself
        std::mem::forget(self);

        Ok(RecordingCommandBuffer {
            parent_pool,
            inner: cb,
            ended: false,
        })
    }

    /// Begin recording with ONE_TIME_SUBMIT usage flag (convenience).
    pub fn begin_one_time(self) -> Result<RecordingCommandBuffer, vk::Result> {
        self.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        let pool = self.parent_pool.pool.lock().ignore_poison();
        //SAFETY: Exclusive access to pool due to lock, command buffer was
        //allocated on this pool, pool was created on parent device
        unsafe {
            self.parent_pool
                .parent_device
                .inner()
                .free_command_buffers(*pool, &[self.inner])
        };
    }
}

/// Represents a command buffer in the RECORDING state. This is returned by
/// `CommandBuffer::begin` and must be ended via `RecordingCommandBuffer::end`
/// to return a usable `CommandBuffer`.
#[derive(Debug)]
pub struct RecordingCommandBuffer {
    parent_pool: Arc<CommandPool>,
    inner: vk::CommandBuffer,
    ended: bool,
}

impl RecordingCommandBuffer {
    /// Get the raw `vk::CommandBuffer` handle while recording.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.inner
    }

    /// Transition the swapchain image into a layout suitable for color
    /// attachment rendering (COLOR_ATTACHMENT_OPTIMAL).
    pub fn transition_swapchain_image_for_rendering(
        &mut self,
        swapchain: &crate::rvk::swapchain::Swapchain,
        image_index: u32,
    ) {
        let _device = self.parent_pool.parent_device.inner();
        let img = swapchain.image(image_index as usize);

        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(img)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        unsafe {
            self.parent_pool.parent_device.inner().cmd_pipeline_barrier(
                self.inner,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
    }

    /// Clear the color of the given swapchain image (assumes it's in
    /// COLOR_ATTACHMENT_OPTIMAL layout).
    pub fn clear_color_attachment(
        &mut self,
        swapchain: &crate::rvk::swapchain::Swapchain,
        image_index: u32,
        color: [f32; 4],
    ) {
        let img = swapchain.image(image_index as usize);
        let clear_value = vk::ClearColorValue { float32: color };

        unsafe {
            self.parent_pool
                .parent_device
                .inner()
                .cmd_clear_color_image(
                    self.inner,
                    img,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    &clear_value,
                    &[vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)],
                )
        };
    }

    /// Transition the swapchain image into PRESENT_SRC_KHR for presentation.
    pub fn transition_swapchain_image_for_presenting(
        &mut self,
        swapchain: &crate::rvk::swapchain::Swapchain,
        image_index: u32,
    ) {
        let img = swapchain.image(image_index as usize);
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(img)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        unsafe {
            self.parent_pool.parent_device.inner().cmd_pipeline_barrier(
                self.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
    }

    /// Begin dynamic rendering with a single color attachment and clear it to
    /// `clear_color`. Returns a small helper which allows setting dynamic
    /// state (viewport/scissor), binding pipelines and issuing draw calls.
    /// The returned helper will end the dynamic rendering in its Drop impl.
    pub fn begin_rendering_clear_color<'a>(
        &'a mut self,
        swapchain: &crate::rvk::swapchain::Swapchain,
        image_index: u32,
        clear_color: vek::Vec4<f32>,
    ) -> InlineRenderer<'a> {
        use ash::vk::{
            self, ClearColorValue, ClearValue, Rect2D, RenderingAttachmentInfo, RenderingInfo,
        };

        let device = self.parent_pool.parent_device.clone();
        let image_view = swapchain.image_view(image_index as usize);

        let clear_value = ClearValue {
            color: ClearColorValue {
                float32: clear_color.into_array(),
            },
        };

        let attachment = RenderingAttachmentInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);

        let render_area = Rect2D::default().extent(vk::Extent2D {
            width: swapchain.width(),
            height: swapchain.height(),
        });

        let attachments = [attachment];
        let rendering_info = RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&attachments);

        // SAFETY: we recorded a valid command buffer and we are using the
        // device that created it.
        unsafe {
            device
                .inner()
                .cmd_begin_rendering(self.inner, &rendering_info)
        };

        InlineRenderer {
            device,
            cb: self.inner,
            _lifetime: std::marker::PhantomData,
        }
    }

    /// End recording and return a `RecordedCommandBuffer` representing the
    /// recorded command buffer ready for submission.
    pub fn end(mut self) -> Result<RecordedCommandBuffer, vk::Result> {
        let device = self.parent_pool.parent_device.inner();
        // SAFETY: ending a valid recording is a valid operation
        unsafe { device.end_command_buffer(self.inner) }?;
        self.ended = true;
        Ok(RecordedCommandBuffer(CommandBuffer {
            parent_pool: self.parent_pool.clone(),
            inner: self.inner,
        }))
    }
}

impl Drop for RecordingCommandBuffer {
    fn drop(&mut self) {
        if !self.ended {
            // Best-effort end; log but do not panic on failure.
            let device = self.parent_pool.parent_device.inner();
            let res = unsafe { device.end_command_buffer(self.inner) };
            if let Err(e) = res {
                tracing::warn!("drop: failed to end recording command buffer: {:?}", e);
            }
        }
    }
}
/// Helper representing an active dynamic rendering region. Drop will call
/// `cmd_end_rendering` on the command buffer.
#[derive(Debug)]
pub struct InlineRenderer<'a> {
    device: std::sync::Arc<crate::rvk::device::Device>,
    cb: vk::CommandBuffer,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl<'a> InlineRenderer<'a> {
    pub fn set_scissor(&self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };
        unsafe { self.device.inner().cmd_set_scissor(self.cb, 0, &[scissor]) };
    }

    pub fn set_viewport(
        &self,

        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        let vp = vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        };
        unsafe { self.device.inner().cmd_set_viewport(self.cb, 0, &[vp]) };
    }

    pub fn bind_pipeline(&self, pipeline: &crate::rvk::pipeline::DynamicPipeline) {
        unsafe {
            self.device.inner().cmd_bind_pipeline(
                self.cb,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw(),
            )
        };
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.inner().cmd_draw(
                self.cb,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };
    }
}

impl<'a> Drop for InlineRenderer<'a> {
    fn drop(&mut self) {
        unsafe { self.device.inner().cmd_end_rendering(self.cb) };
    }
}

/// Newtype representing a command buffer which has finished recording and is
/// ready for submission. Wraps the owned `CommandBuffer` returned after
/// ending a recording.
#[derive(Debug)]
pub struct RecordedCommandBuffer(pub CommandBuffer);

impl RecordedCommandBuffer {
    /// Get raw vk handle of the recorded command buffer.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.0.raw()
    }

    /// Consume the newtype and return the inner `CommandBuffer` for manual
    /// operations or freeing.
    pub fn into_inner(self) -> CommandBuffer {
        self.0
    }
}
