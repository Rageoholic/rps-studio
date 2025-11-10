#![allow(dead_code, unused_variables, reason = "In work")]

use std::sync::{Arc, LockResult, Mutex};

/// Extension methods for [`LockResult`].
///
/// [`LockResult`]: https://doc.rust-lang.org/stable/std/sync/type.LockResult.html
pub trait LockResultExt {
    type Guard;

    /// Returns the lock guard even if the mutex is [poisoned].
    ///
    /// [poisoned]: https://doc.rust-lang.org/stable/std/sync/struct.Mutex.html#poisoning
    fn ignore_poison(self) -> Self::Guard;
}

impl<Guard> LockResultExt for LockResult<Guard> {
    type Guard = Guard;

    fn ignore_poison(self) -> Guard {
        self.unwrap_or_else(|e| e.into_inner())
    }
}

fn main() {
    let x = Arc::new(Mutex::new(0));
    println!("{}", x.lock().ignore_poison());
}

///Pipeline related functionality
pub mod pipeline {
    use std::sync::Arc;

    use ash::vk::{self};
    use thiserror::Error;

    use crate::rvk::{shader::Shader, swapchain::Swapchain};

    use super::device::Device;

    ///Represents an RAII vk::PipelineLayout
    #[derive(Debug)]
    pub struct PipelineLayout {
        parent: Arc<Device>,
        inner: vk::PipelineLayout,
    }

    impl Drop for PipelineLayout {
        fn drop(&mut self) {
            // SAFETY: inner was derived from parent
            unsafe {
                self.parent
                    .inner()
                    .destroy_pipeline_layout(self.inner, None)
            };
        }
    }

    #[derive(Debug, Error)]
    pub enum PipelineLayoutCreateError {
        #[error("Unknown vulkan error {0}")]
        UnknownVulkan(vk::Result),
    }

    impl PipelineLayout {
        pub fn new(device: &Arc<Device>) -> Result<Self, PipelineLayoutCreateError> {
            let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default();

            // SAFETY: we're passing in a valid ci
            let pipeline_layout = unsafe {
                device
                    .inner()
                    .create_pipeline_layout(&pipeline_layout_ci, None)
            }
            .map_err(PipelineLayoutCreateError::UnknownVulkan)?;
            Ok(Self {
                parent: device.clone(),
                inner: pipeline_layout,
            })
        }
    }

    #[derive(Debug)]
    pub struct DynamicPipeline {
        parent: Arc<Device>,
        _layout: Arc<PipelineLayout>,
        _vert_shader: Arc<Shader>,
        _frag_shader: Arc<Shader>,
        pipeline: vk::Pipeline,
    }

    #[derive(Debug, Error)]
    pub enum PipelineCreateError {
        #[error("You passed in parameters that aren't all from the same vulkan device")]
        MismatchedDevices,
        #[error("Unknown vk error: {0}")]
        UnknownVk(vk::Result),
    }

    impl DynamicPipeline {
        pub fn new(
            parent_device: &Arc<Device>,
            layout: &Arc<PipelineLayout>,
            parent_swapchain: &Arc<Swapchain>,
            vert_shader: &Arc<Shader>,
            frag_shader: &Arc<Shader>,
        ) -> Result<Self, PipelineCreateError> {
            use PipelineCreateError as Error;
            if **parent_device != *Swapchain::get_parent(parent_swapchain) {
                Err(Error::MismatchedDevices)?
            }
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
            //TODO: parse vertex state passed to us
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let color_blend_states = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)];
            let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(&color_blend_states);

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .module((vert_shader).inner())
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .module(frag_shader.inner())
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(c"main"),
            ];

            let color_attachment_formats = [parent_swapchain.get_color_format()];
            let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&color_attachment_formats);

            let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_ci)
                .dynamic_state(&dynamic_state)
                .layout(layout.inner)
                .subpass(0)
                .push_next(&mut rendering_info);

            //SAFETY: valid ci
            let pipeline = unsafe {
                parent_device.inner().create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_ci],
                    None,
                )
            }
            .map_err(|e| {
                for p in e.0 {
                    //SAFETY: created p from parent
                    unsafe { parent_device.inner().destroy_pipeline(p, None) };
                }
                Error::UnknownVk(e.1)
            })?;

            Ok(Self {
                parent: parent_device.clone(),
                _layout: layout.clone(),

                _vert_shader: vert_shader.clone(),
                _frag_shader: frag_shader.clone(),
                pipeline: pipeline[0],
            })
        }
    }

    impl Drop for DynamicPipeline {
        fn drop(&mut self) {
            //SAFETY: pipeline is from parent
            unsafe { self.parent.inner().destroy_pipeline(self.pipeline, None) };
        }
    }
}

pub mod command_buffers {
    use std::{
        fmt::Display,
        sync::{Arc, Mutex},
    };

    use ash::vk::{self, Handle};
    use thiserror::Error;

    use crate::rvk::{device::Device, LockResultExt};

    #[derive(Debug)]
    pub(crate) struct CommandPool {
        parent_device: Arc<Device>,
        pool: Mutex<vk::CommandPool>,
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
        ) -> Result<Vec<RaiiCommandBuffer>, AllocateCommandBuffersError> {
            use AllocateCommandBuffersError as Error;
            use AllocateCommandBuffersErrorType as ErrorType;
            // Get exclusive access to the pool
            let pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let pool = *pool;
            let cbai = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(count);

            //SAFETY: pool comes from parent device, have exclusive access,
            //valid allocate_info
            let cbs = unsafe { self.parent_device.inner().allocate_command_buffers(&cbai) }
                .map_err(|e| Error {
                    e: match e {
                        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                        | vk::Result::ERROR_OUT_OF_HOST_MEMORY => ErrorType::MemoryExhaustion,
                        _ => ErrorType::UnknownVk(e),
                    },
                    count,
                    pool,
                })?;

            Ok(cbs
                .iter()
                .map(|raw_cb| RaiiCommandBuffer {
                    parent_pool: self.clone(),
                    inner: *raw_cb,
                })
                .collect())
        }
    }
    impl Drop for CommandPool {
        fn drop(&mut self) {
            //SAFETY: Exclusive access to pool, pool was created from parent_device
            unsafe {
                self.parent_device.inner().destroy_command_pool(
                    *self.pool.get_mut().unwrap_or_else(|e| e.into_inner()),
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
    pub struct UnrecordedCommandBuffer {
        cb: RaiiCommandBuffer,
    }

    #[derive(Debug)]
    pub struct RaiiCommandBuffer {
        parent_pool: Arc<CommandPool>,
        inner: vk::CommandBuffer,
    }

    impl Drop for RaiiCommandBuffer {
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

    #[derive(Debug)]
    pub struct CommandBufferEncoder {
        cb: RaiiCommandBuffer,
    }

    #[derive(Debug, Error)]
    pub enum CommandBufferBeginEncodeError {
        #[error("Unknown Vulkan: {0}")]
        UnknownVk(vk::Result),
        #[error("Memory exhaustion")]
        MemoryExhaustion,
    }

    impl RaiiCommandBuffer {
        pub fn begin_encode(self) -> Result<CommandBufferEncoder, CommandBufferBeginEncodeError> {
            use CommandBufferBeginEncodeError as Error;
            //SAFETY: inner is a valid allocated command buffer, inner is
            //allocated from parent_pool, parent_pool is derived from device
            unsafe {
                self.parent_pool.parent_device.inner().begin_command_buffer(
                    self.inner,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .map_err(|e| match e {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    Error::MemoryExhaustion
                }
                _ => Error::UnknownVk(e),
            })?;
            Ok(CommandBufferEncoder { cb: self })
        }
    }

    #[derive(Debug)]
    pub struct EncodedCommandBuffer {
        cb: RaiiCommandBuffer,
    }

    #[derive(Error, Debug)]
    pub enum FinishEncodingError {
        #[error("Exhausted Memory")]
        MemoryExhaustion,
        #[error("Unknown Vulkan Error {0}")]
        UnknownVulkan(vk::Result),
    }

    impl CommandBufferEncoder {
        pub fn finish(self) -> Result<EncodedCommandBuffer, FinishEncodingError> {
            use FinishEncodingError as Error;

            //SAFETY: Being in a CommandBufferEncoder means that we are
            //recording the command buffer. This ends that recording. self.cb is
            //a valid RaiiCommandBuffer
            unsafe {
                self.cb
                    .parent_pool
                    .parent_device
                    .inner()
                    .end_command_buffer(self.cb.inner)
            }
            .map_err(|e| match e {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                    Error::MemoryExhaustion
                }
                _ => Error::UnknownVulkan(e),
            })?;

            Ok(EncodedCommandBuffer { cb: self.cb })
        }
    }
}

///Functionality related to swapchains
pub mod swapchain {
    use std::{fmt::Debug, sync::Arc};

    use ash::vk::{
        self, ColorSpaceKHR, ComponentMapping, CompositeAlphaFlagsKHR, Extent2D, Format, Image,
        ImageAspectFlags, ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo,
        ImageViewType, PresentModeKHR, SharingMode, SurfaceTransformFlagsKHR,
        SwapchainCreateInfoKHR, SwapchainKHR,
    };

    use crate::rvk::{device::Device, surface::Surface};

    /// Thing that we render to
    pub struct Swapchain {
        /// Device that we were made with
        parent_device: Arc<Device>,
        /// Surface that we were made with
        parent_surface: Arc<Surface>,
        /// Loaded extension function pointers
        swapchain_device: ash::khr::swapchain::Device,
        /// Actual handle to the surface
        swapchain: SwapchainKHR,
        /// Images associated with the swapchain
        _images: Vec<Image>,
        /// Image views associated with above images
        image_views: Vec<ImageView>,
        _surface_format: ash::vk::SurfaceFormatKHR,
    }

    impl PartialEq for Swapchain {
        fn eq(&self, other: &Self) -> bool {
            self.parent_device == other.parent_device
                && self.parent_surface == other.parent_surface
                && self.swapchain == other.swapchain
        }
    }

    #[derive(thiserror::Error, Debug)]
    /// Errors that pop up when we create a swapchain
    pub enum SwapchainCreateError {
        /// We don't know what went wrong but vulkan did it. Effectively the
        /// default shrug answer
        #[error("Unknown Vulkan error {0}")]
        UnknownVulkan(ash::vk::Result),
        /// You passed in a surface and device that are not derived from the
        /// same instance
        #[error("Incompatible parameters (device and surface are not from same instance)")]
        IncompatibleParameters,
        /// For some reason we can't pick a format. So we can't create a swapchain. Should basically
        /// never happen
        #[error("No compatible format found")]
        NoCompatibleFormat,
    }
    impl From<ash::vk::Result> for SwapchainCreateError {
        fn from(value: ash::vk::Result) -> Self {
            Self::UnknownVulkan(value)
        }
    }

    impl Swapchain {
        /// Create a swapchain from this device and surface. Might wanna
        /// make some optional parameters to configure? But rn this is
        /// wrapping how *I* set up a swapchain
        pub(crate) fn create(
            device: &Arc<Device>,
            surface: &Arc<Surface>,
            old_swapchain: Option<&Self>,
        ) -> Result<Swapchain, SwapchainCreateError> {
            if device.get_parent().raw_handle() != surface.parent_instance().raw_handle() {
                return Err(SwapchainCreateError::IncompatibleParameters);
            }
            let swapchain_device =
                ash::khr::swapchain::Device::new(device.parent_instance().inner(), device.inner());

            //SAFETY: surface and device are derived from the same instance
            let (surface_capabilities, surface_formats, present_modes) = unsafe {
                (
                    surface.get_capabilities_unsafe(device)?,
                    surface.get_formats_unsafe(device)?,
                    surface.get_present_modes_unsafe(device)?,
                )
            };

            let present_mode = present_modes
                .iter()
                .find(|pm| **pm == PresentModeKHR::MAILBOX)
                .copied()
                .unwrap_or(PresentModeKHR::FIFO);

            let surface_format = surface_formats
                .iter()
                .find(|format| {
                    format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                        && format.format == Format::B8G8R8A8_SRGB
                })
                .copied()
                .or(surface_formats.first().copied())
                .ok_or(SwapchainCreateError::NoCompatibleFormat)?;

            let swapchain_extent = if surface_capabilities.current_extent.width != u32::MAX {
                surface_capabilities.current_extent
            } else {
                let surface_extent = surface.get_surface_extent();

                Extent2D {
                    width: surface_extent.width.clamp(
                        surface_capabilities.min_image_extent.width,
                        surface_capabilities.max_image_extent.height,
                    ),
                    height: surface_extent.height.clamp(
                        surface_capabilities.min_image_extent.height,
                        surface_capabilities.max_image_extent.height,
                    ),
                }
            };
            let default_image_count = surface_capabilities.min_image_count + 1;
            let max_image_count = surface_capabilities.max_image_count;
            let requested_image_count =
                if max_image_count != 0 && max_image_count < default_image_count {
                    max_image_count
                } else {
                    default_image_count
                };

            //TODO: Different image sharing mode?
            let ci = SwapchainCreateInfoKHR::default()
                .present_mode(present_mode)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(swapchain_extent)
                .min_image_count(requested_image_count)
                .image_array_layers(1)
                .image_sharing_mode(SharingMode::EXCLUSIVE)
                .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                .surface(surface.inner())
                .pre_transform(SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                .old_swapchain(
                    old_swapchain
                        .map(|s| s.swapchain)
                        .unwrap_or(vk::SwapchainKHR::null()),
                );

            //SAFETY: Valid ci
            let swapchain = unsafe { swapchain_device.create_swapchain(&ci, None) }?;

            //SAFETY: Swapchain was made from this swapchain_device
            let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }?;

            let mut image_views = Vec::with_capacity(images.len());

            for image in &images {
                let ivci = ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(ImageViewType::TYPE_2D)
                    .components(ComponentMapping::default())
                    .format(surface_format.format)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                //SAFETY: ivci.image was created from the swapchain which is
                //derived from device
                let iv = unsafe { device.inner().create_image_view(&ivci, None) }?;
                image_views.push(iv);
            }

            Ok(Self {
                parent_device: device.clone(),
                parent_surface: surface.clone(),
                swapchain,
                swapchain_device,
                _images: images,
                image_views,
                _surface_format: surface_format,
            })
        }

        pub(crate) fn get_color_format(&self) -> vk::Format {
            self._surface_format.format
        }

        pub(crate) fn get_parent(self: &Swapchain) -> &Device {
            &self.parent_device
        }
    }

    impl Debug for Swapchain {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Swapchain")
                .field("parent_device", &self.parent_device)
                .field("parent_surface", &self.parent_surface)
                .field("swapchain_device", &"{opaque}")
                .field("swapchain", &self.swapchain)
                .finish()
        }
    }

    impl Drop for Swapchain {
        fn drop(&mut self) {
            for iv in self.image_views.drain(..) {
                //SAFETY: iv was created on this device
                unsafe { self.parent_device.inner().destroy_image_view(iv, None) };
            }
            //SAFETY: Swapchain was made with this swapchain_device
            unsafe {
                self.swapchain_device
                    .destroy_swapchain(self.swapchain, None)
            };
        }
    }
}
/// Functionality related to the instance. Exists for scoping reasons
pub mod instance {
    use ash::{
        vk::{
            self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
            DebugUtilsMessengerEXT,
        },
        Entry,
    };
    use debug_messenger::instance_debug_callback;
    use std::{ffi::CStr, fmt::Debug};
    use winit::raw_window_handle::DisplayHandle;

    use crate::Version;

    /// Represents a VkInstance and contains the Entry for proper management
    /// protocols. Also bundles the debug messenger.
    pub struct Instance {
        /// Represents the entry point functions as well as an owned pointer to the
        /// underlying vulkan shared library.
        entry: ash::Entry,
        /// Represents a VkInstance and the corresponding function pointers
        instance: ash::Instance,
        /// Represents a VkDebugUtilsMessengerEXT that may or may not be present
        debug_messenger: Option<DebugMessenger>,
    }

    impl PartialEq for Instance {
        fn eq(&self, other: &Self) -> bool {
            self.instance.handle() == other.instance.handle()
                && self.debug_messenger == other.debug_messenger
        }
    }
    impl Eq for Instance {}
    impl Drop for Instance {
        fn drop(&mut self) {
            if let Some(dm) = self.debug_messenger.take() {
                //SAFETY: Last usage of debug_messenger. This deinitializes it.
                unsafe {
                    dm.debug_utils_instance
                        .destroy_debug_utils_messenger(dm.debug_messenger, None)
                };
            };
            //SAFETY: Last use of Instance, all children destroyed
            unsafe { self.instance.destroy_instance(None) };
        }
    }
    impl Debug for Instance {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Instance")
                .field("entry", &"{entry}")
                .field("instance", &self.instance.handle())
                .finish()
        }
    }

    #[derive(Debug, thiserror::Error)]
    /// Possible errors returned by Instance::new
    pub enum InstanceCreateError {
        #[error("Unspecified vk error: {0}")]
        /// Error has not been assigned a specific error variant. Some vk function
        /// just done fucked up in an unforseen way
        UnspecifiedVulkan(ash::vk::Result),
        #[error("Error loading vulkan: {0}")]
        /// Could not load our vulkan entry points from ash.
        EntryLoading(ash::LoadingError),
        #[error("Missing necessary window extensions")]
        /// This vulkan impl does not implement the necessary extensions for
        /// rendering (VK_SURFACE_KHR and the corresponding platform extension to
        /// get a VkSurface)
        MissingWindowingExtensions,
    }
    impl From<ash::vk::Result> for InstanceCreateError {
        fn from(value: ash::vk::Result) -> Self {
            Self::UnspecifiedVulkan(value)
        }
    }

    impl Instance {
        /// Get a reference to the associated debug messenger
        pub(super) fn get_debug_messenger(&self) -> Option<&DebugMessenger> {
            self.debug_messenger.as_ref()
        }
        /// Get a ref to the ash::Entry associated with this instance
        pub(super) fn parent(&self) -> &Entry {
            &self.entry
        }
        /// Get the Instance handle for comparing with other handles
        pub(super) fn raw_handle(&self) -> ash::vk::Instance {
            self.inner().handle()
        }
        /// Get the ash Instance for passing to ash
        pub(super) fn inner(&self) -> &ash::Instance {
            &self.instance
        }
        /// Construct an instance. Debug Utils validation will be initialized to debug_level if possible
        pub fn new(
            minimum_vk_version: Version,
            debug_level: VulkanDebugLevel,
            display_handle: DisplayHandle,
        ) -> Result<Self, InstanceCreateError> {
            use InstanceCreateError as Error;
            //SAFETY: Somewhat inherently unsafe due to loading a shared library
            //which can run arbitrary initialization code. However, we are probably
            //fine since this is the vulkan lib which means it *shouldn't* fuck with
            //arbitrary memory or something stupid like that
            let entry = unsafe { ash::Entry::load() }.map_err(Error::EntryLoading)?;

            //TODO: Engine version, app name, app version
            let app_info = ash::vk::ApplicationInfo::default()
                .api_version(minimum_vk_version.to_vk_version())
                .engine_name(c"RPS Studio");

            let available_extensions_vec = {
                //SAFETY: Basically always safe because of ash. Only marked unsafe
                //because it's technically an FFI call wrapper but ash manages the
                //memory management for us
                unsafe { entry.enumerate_instance_extension_properties(None) }?
            };

            let available_extensions = available_extensions_vec
                .iter()
                .map(|ext| ext.extension_name_as_c_str().unwrap_or(c""));
            //SAFETY: Basically always safe
            let available_layers_vec = unsafe { entry.enumerate_instance_layer_properties() }?;

            let available_layers = available_layers_vec
                .iter()
                .map(|layer| layer.layer_name_as_c_str().unwrap_or(c""));

            let window_system_exts =
                ash_window::enumerate_required_extensions(display_handle.as_raw())?;

            if !window_system_exts
                .iter()
                .cloned()
                .map(|ptr|
                        //SAFETY: ash_window has to give us valid C Strings or the
                        //function is useless
                        unsafe { CStr::from_ptr(ptr) })
                .all(|ext_name| available_extensions.clone().any(|ext| ext.eq(ext_name)))
            {
                return Err(Error::MissingWindowingExtensions);
            }

            const VK_KHRONOS_VALIDATION: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

            let mut enabled_extension_names = Vec::with_capacity(8);

            let mut enabled_layer_names = Vec::with_capacity(1);

            enabled_extension_names.extend_from_slice(window_system_exts);

            let debug_enabled = debug_level != VulkanDebugLevel::None
                && available_extensions
                    .clone()
                    .any(|ext| ext.eq(ash::ext::debug_utils::NAME))
                && available_layers
                    .clone()
                    .any(|ext| ext.eq(VK_KHRONOS_VALIDATION));

            let mut debug_ci = debug_enabled.then(|| {
                enabled_extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
                enabled_layer_names.push(VK_KHRONOS_VALIDATION.as_ptr());
                ash::vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(debug_level.sev_flags())
                    .message_type(
                        DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                            | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    )
                    .pfn_user_callback(Some(instance_debug_callback))
            });

            let mut ci = ash::vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&enabled_extension_names)
                .enabled_layer_names(&enabled_layer_names);

            if let Some(dci) = debug_ci.as_mut() {
                ci = ci.push_next(dci)
            }

            //SAFETY: Valid InstanceCI being used, Instance is destroyed before
            //entry
            let instance = unsafe { entry.create_instance(&ci, None) }?;

            let debug_messenger = if let Some(debug_ci) = debug_ci.as_ref() {
                let debug_utils_instance = ash::ext::debug_utils::Instance::new(&entry, &instance);
                //SAFETY: We follow all of vulkan's rules
                let debug_messenger =
                    unsafe { debug_utils_instance.create_debug_utils_messenger(debug_ci, None) }?;
                Some(DebugMessenger {
                    debug_messenger,
                    debug_utils_instance,
                })
            } else {
                None
            };

            Ok(Self {
                entry,
                instance,
                debug_messenger,
            })
        }

        /// Get a Vec of all physical devices available to this instance
        pub(super) fn enumerate_physical_devices(
            &self,
        ) -> ash::prelude::VkResult<Vec<vk::PhysicalDevice>> {
            // SAFETY: Effectively made safe by ash managing the memory for us
            unsafe { self.instance.enumerate_physical_devices() }
        }
    }

    /// Functionality related to the debug manager
    mod debug_messenger {
        use std::ffi::c_void;

        use ash::vk::{
            Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
            DebugUtilsMessengerCallbackDataEXT,
        };

        /// Debug Utils Messenger callback. Not for external calling
        pub(super) unsafe extern "system" fn instance_debug_callback(
            message_severity: DebugUtilsMessageSeverityFlagsEXT,
            message_types: DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: *const DebugUtilsMessengerCallbackDataEXT<'_>,
            _: *mut c_void,
        ) -> Bool32 {
            //SAFETY: Vulkan has to give us a valid callback data
            let callback_data = unsafe { *p_callback_data };

            let msg_id = callback_data.message_id_number;
            //SAFETY: Vulkan must give us a valid message name
            let msg_name = unsafe { callback_data.message_id_name_as_c_str() }.unwrap();
            //SAFETY: Vulkan must give us a valid message
            let msg_data = unsafe { callback_data.message_as_c_str() }.unwrap();
            let msg_name = msg_name.to_str().unwrap();
            let msg_data = msg_data.to_str().unwrap();
            use DebugUtilsMessageSeverityFlagsEXT as SevFlags;
            use DebugUtilsMessageTypeFlagsEXT as TyFlags;
            let mut ty_name = String::with_capacity(16);
            if message_types.contains(TyFlags::DEVICE_ADDRESS_BINDING) {
                ty_name.push_str("ADDRESS BINDING");
            }
            if message_types.contains(TyFlags::GENERAL) {
                ty_name.push_str("GENERAL");
            }
            if message_types.contains(TyFlags::PERFORMANCE) {
                ty_name.push_str("PERF");
            }
            if message_types.contains(TyFlags::VALIDATION) {
                ty_name.push_str("VALIDATION");
            }
            if message_severity.contains(SevFlags::ERROR) {
                tracing::error!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
            } else if message_severity.contains(SevFlags::WARNING) {
                tracing::warn!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
            } else if message_severity.contains(SevFlags::INFO) {
                tracing::info!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
            } else if message_severity.contains(SevFlags::VERBOSE) {
                tracing::trace!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
            }

            ash::vk::FALSE
        }
    }
    /// NONOWNING implementation of a Debug Messenger. Meant only to be used in conjunction with Instance
    pub(super) struct DebugMessenger {
        /// The inner debug_messenger handle
        debug_messenger: DebugUtilsMessengerEXT,
        /// The loaded extension function pointers and stuff
        debug_utils_instance: ash::ext::debug_utils::Instance,
    }
    impl PartialEq for DebugMessenger {
        fn eq(&self, other: &Self) -> bool {
            self.debug_messenger == other.debug_messenger
        }
    }
    impl Eq for DebugMessenger {}

    #[derive(
        Debug,
        Default,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        strum_macros::EnumString,
        strum_macros::Display,
        Clone,
        Copy,
    )]

    /// Represents a VkDebugUtilsMessageSeverityFlags option, representing itself
    /// and every option below it
    pub(crate) enum VulkanDebugLevel {
        #[default]
        /// No debug layer messages
        None,
        /// Print out all messages
        Verbose,
        /// Print out useful information
        Info,
        /// Print out warnings for misuse of certain APIs
        Warn,
        /// Print out only errors like failing to destroy child objects
        Error,
    }
    impl VulkanDebugLevel {
        /// Convert to the corresponding VkDebugUtilsMessageSeverityFlagsEXT
        fn sev_flags(&self) -> DebugUtilsMessageSeverityFlagsEXT {
            match self {
                VulkanDebugLevel::None => DebugUtilsMessageSeverityFlagsEXT::empty(),
                VulkanDebugLevel::Verbose => {
                    DebugUtilsMessageSeverityFlagsEXT::VERBOSE | Self::Info.sev_flags()
                }
                VulkanDebugLevel::Info => {
                    DebugUtilsMessageSeverityFlagsEXT::INFO | Self::Warn.sev_flags()
                }
                VulkanDebugLevel::Warn => {
                    DebugUtilsMessageSeverityFlagsEXT::WARNING | Self::Error.sev_flags()
                }
                VulkanDebugLevel::Error => DebugUtilsMessageSeverityFlagsEXT::ERROR,
            }
        }
    }
}

/// Functionality related to the surface. Exists for scoping reasons
pub mod surface {
    use std::{fmt::Debug, sync::Arc};

    use ash::{
        prelude::VkResult,
        vk::{self, Extent2D, PresentModeKHR, SurfaceFormatKHR, SurfaceKHR},
    };
    use thiserror::Error;
    use winit::{
        raw_window_handle::{HasDisplayHandle, HasWindowHandle},
        window::Window,
    };

    use super::{device::Device, instance::Instance};

    /// Represents a VkSurfaceKHR.
    pub(crate) struct Surface {
        /// Loaded function pointers for VK_SURFACE_KHR
        surface_instance: ash::khr::surface::Instance,
        /// The underlying VkSurfaceKHR
        surface: SurfaceKHR,
        /// Reference counted pointer to the instance. Here for RAII purposes as
        /// Instance must be destroyed *after* surface
        parent_instance: Arc<Instance>,
        /// Reference counted pointer to the underlying window. Here for RAII
        /// purposes as Window must be destroyed *after* surface
        parent_window: Arc<Window>,
    }

    impl PartialEq for Surface {
        fn eq(&self, other: &Self) -> bool {
            self.surface == other.surface
                && self.parent_instance == other.parent_instance
                && self.parent_window.id() == other.parent_window.id()
        }
    }
    impl Eq for Surface {}

    impl Debug for Surface {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Surface")
                .field("surface_instance", &"...")
                .field("surface", &self.surface)
                .field("_parent_instance", &self.parent_instance)
                .field("_parent_window", &self.parent_window)
                .finish()
        }
    }

    impl Drop for Surface {
        fn drop(&mut self) {
            //SAFETY: Last use of surface, all children destroyed
            unsafe { self.surface_instance.destroy_surface(self.surface, None) };
        }
    }

    #[derive(Debug, Error)]
    /// The set of possible errors when creating a Surface
    pub enum SurfaceCreateError {
        #[error("Unknown Vulkan Error {0}")]
        /// An error that has not yet been defined
        UnknownVulkan(ash::vk::Result),
        /// Couldn't get a display handle
        #[error("Couldn't get display handle")]
        InvalidDisplayHandle,
        /// Couldn't get a window handle
        #[error("Couldn't get window handle")]
        InvalidWindowHandle,
    }

    impl From<ash::vk::Result> for SurfaceCreateError {
        fn from(value: ash::vk::Result) -> Self {
            SurfaceCreateError::UnknownVulkan(value)
        }
    }

    impl Surface {
        ///Get reference to the parent instance
        pub(super) fn parent_instance(&self) -> &Arc<Instance> {
            &self.parent_instance
        }

        ///Get a handle to the inner surface for whenever necessary
        pub(super) fn inner(&self) -> ash::vk::SurfaceKHR {
            self.surface
        }
        /// Get the extent of the surface in Vulkan terms
        pub fn get_surface_extent(&self) -> Extent2D {
            let inner_size = self.parent_window.inner_size();
            Extent2D {
                width: inner_size.width,
                height: inner_size.height,
            }
        }

        /// Get the formats associated with this surface on this device.
        ///
        /// # SAFETY
        ///
        /// Surface and Device are from the same instance
        pub unsafe fn get_formats_unsafe(
            &self,
            device: &Device,
        ) -> VkResult<Vec<SurfaceFormatKHR>> {
            //SAFETY: We pass the responsibility for the Device being from
            //the same instance as us to the caller
            unsafe {
                self.surface_instance
                    .get_physical_device_surface_formats(device.get_phys_dev(), self.surface)
            }
        }

        /// Gets the present modes associated with this surface on this device
        ///
        /// # SAFETY
        ///
        /// Surface and Device are from the same instance
        pub unsafe fn get_present_modes_unsafe(
            &self,
            device: &Device,
        ) -> VkResult<Vec<PresentModeKHR>> {
            //SAFETY: We pass the responsibility for the Device being from
            //the same instance as us to the caller
            unsafe {
                self.surface_instance
                    .get_physical_device_surface_present_modes(device.get_phys_dev(), self.surface)
            }
        }

        /// Checks if queue_family_index associated with the physical_device
        /// supports presenting to this Surface.
        ///
        /// # SAFETY
        ///
        /// Surface and Device are from the same instance
        pub unsafe fn _get_support_unsafe(
            &self,
            device: &Device,
            queue_family_index: u32,
        ) -> VkResult<bool> {
            //SAFETY: Surface and Device are from the same instance
            unsafe {
                self.surface_instance.get_physical_device_surface_support(
                    device.get_phys_dev(),
                    queue_family_index,
                    self.surface,
                )
            }
        }

        /// Get the capabilities associated with this surface on this device.
        ///
        /// # SAFETY
        ///
        /// Surface and Device are from the same instance
        pub unsafe fn get_capabilities_unsafe(
            &self,
            device: &Device,
        ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
            //SAFETY: We pass the responsibility for the Device being from
            //the same instance as us to the caller
            unsafe {
                self.surface_instance
                    .get_physical_device_surface_capabilities(device.get_phys_dev(), self.surface)
            }
        }
        /// Uses a winit Window in order to create a surface
        pub(crate) fn from_winit_window(
            win: &Arc<Window>,
            instance: &Arc<Instance>,
        ) -> Result<Self, SurfaceCreateError> {
            let surface_instance =
                ash::khr::surface::Instance::new(instance.parent(), instance.inner());
            //SAFETY: Passing a valid window and display handle. Enforces
            //the parent/child relationship between the surface and the
            //instance/window by holding onto an Arc to the instance/window
            let surface = unsafe {
                ash_window::create_surface(
                    instance.parent(),
                    instance.inner(),
                    win.display_handle()
                        .map_err(|_| SurfaceCreateError::InvalidDisplayHandle)?
                        .as_raw(),
                    win.window_handle()
                        .map_err(|_| SurfaceCreateError::InvalidWindowHandle)?
                        .as_raw(),
                    None,
                )
            }?;

            Ok(Self {
                surface_instance,
                surface,
                parent_instance: instance.clone(),
                parent_window: win.clone(),
            })
        }
        /// Return a ref to the surface instance
        pub(crate) fn surface_instance(&self) -> &ash::khr::surface::Instance {
            &self.surface_instance
        }
    }
}

/// implementation related to device. Exists for scoping reasons
pub mod device {
    use std::{fmt::Debug, sync::Arc};

    use ash::vk::{
        self, DeviceCreateInfo, DeviceQueueCreateInfo, PhysicalDeviceFeatures2,
        PhysicalDeviceVulkan12Features, PhysicalDeviceVulkan13Features, QueueFlags,
    };
    use thiserror::Error;

    use crate::Version;

    use super::{instance::Instance, surface::Surface};

    //TODO: Support separate present and graphics queue
    /// Represents a VkDevice
    pub struct Device {
        /// Parent for RAII purposes
        parent: Arc<Instance>,
        ///Underlying VkDevice
        device: ash::Device,
        /// A set of function pointers for interacting with the
        /// DebugMessenger in parent
        debug_utils_fps: Option<ash::ext::debug_utils::Device>,
        /// Handle to the physical device. Need to keep it around for some
        /// stuff
        phys_dev: vk::PhysicalDevice,
        /// Handle to the queue we will submit graphics to
        _graphics_queue: vk::Queue,
        /// Index of the graphics queue family
        graphics_queue_family_index: u32,
    }

    impl PartialEq for Device {
        fn eq(&self, other: &Self) -> bool {
            let parent_eq = self.parent == other.parent;
            let handle_eq = self.device.handle() == other.device.handle();
            let phys_dev_eq = self.phys_dev == other.phys_dev;
            let gq_eq = self._graphics_queue == other._graphics_queue;
            let gqi_eq = self.graphics_queue_family_index == other.graphics_queue_family_index;
            parent_eq && handle_eq && phys_dev_eq && gq_eq && gqi_eq
        }
    }

    impl Debug for Device {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Device")
                .field("parent", &self.parent)
                .field("device", &self.device.handle())
                .field(
                    "debug_utils_fps",
                    &if self.debug_utils_fps.is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                )
                .finish()
        }
    }

    /// Errors that can pop up when creating devices
    #[derive(Error, Debug)]
    pub enum DeviceCreateError {
        #[error("Unknown vulkan error {0}")]
        /// An error from vulkan that has not yet been categorized
        UnknownVulkan(ash::vk::Result),
        #[error("Surface was not created from Instance")]
        /// You passed in a Surface that wasn't created from Instance
        InstanceSurfaceMismatch,
        #[error("No suitable device found")]
        /// When we went to scan for devices, we didn't find one that
        /// fulfilled our requirements
        NoSuitableDevice,
    }

    impl From<ash::vk::Result> for DeviceCreateError {
        fn from(value: ash::vk::Result) -> Self {
            Self::UnknownVulkan(value)
        }
    }

    #[derive(Debug)]
    #[expect(dead_code, reason = "We aren't using all of these members yet")]
    #[expect(clippy::missing_docs_in_private_items, reason = "Christ this is a lot")]
    /// Physical device that we've evaluated
    struct ScoredPhysDev<'a> {
        phys_dev: vk::PhysicalDevice,
        score: u32,
        features_11: vk::PhysicalDeviceVulkan11Features<'a>,
        ext_props: Vec<vk::ExtensionProperties>,
        features_10: vk::PhysicalDeviceFeatures,
        features_12: vk::PhysicalDeviceVulkan12Features<'a>,
        features_13: vk::PhysicalDeviceVulkan13Features<'a>,
        props_10: vk::PhysicalDeviceProperties,
        props_11: vk::PhysicalDeviceVulkan11Properties<'a>,
        props_12: vk::PhysicalDeviceVulkan12Properties<'a>,
        props_13: vk::PhysicalDeviceVulkan13Properties<'a>,
        surface_capabilities: Result<vk::SurfaceCapabilitiesKHR, vk::Result>,
        graphics_qfi: u32,
        present_modes: Result<Vec<vk::PresentModeKHR>, vk::Result>,
        formats: Vec<vk::SurfaceFormatKHR>,
    }
    impl Device {
        pub fn get_graphics_queue_family_index(&self) -> u32 {
            self.graphics_queue_family_index
        }
        ///Get a reference to the parent instance
        pub(super) fn get_parent(&self) -> &Instance {
            &self.parent
        }

        ///Get the instance that was used to create this device
        pub(super) fn parent_instance(&self) -> &Arc<Instance> {
            &self.parent
        }

        ///Get the inner ash device for passing into other functions
        pub(super) fn inner(&self) -> &ash::Device {
            &self.device
        }

        /// Create a device that is compatible with the surface
        pub fn create_compatible(
            instance: &Arc<Instance>,
            surface: &Surface,
            min_api_version: Version,
        ) -> Result<Self, DeviceCreateError> {
            use DeviceCreateError as Error;
            if !Arc::ptr_eq(instance, surface.parent_instance()) {
                return Err(Error::InstanceSurfaceMismatch);
            }
            let phys_devs = instance.enumerate_physical_devices()?;
            let scored_phys_dev: ScoredPhysDev =
                match phys_devs.into_iter().fold(None, |best_so_far, current| {
                    // SAFETY: current is a VkPhysicalDevice derived from
                    // instance
                    let ext_props = match unsafe {
                        instance
                            .inner()
                            .enumerate_device_extension_properties(current)
                    } {
                        Ok(exts) => exts,
                        Err(_) => {
                            return best_so_far;
                        }
                    };

                    let mut props_11 = vk::PhysicalDeviceVulkan11Properties::default();
                    let mut props_12 = vk::PhysicalDeviceVulkan12Properties::default();
                    let mut props_13 = vk::PhysicalDeviceVulkan13Properties::default();
                    let mut props = vk::PhysicalDeviceProperties2::default()
                        .push_next(&mut props_11)
                        .push_next(&mut props_12)
                        .push_next(&mut props_13);

                    // SAFETY: current is derived from instance. Valid
                    // current_dev_props
                    unsafe {
                        instance
                            .inner()
                            .get_physical_device_properties2(current, &mut props)
                    };

                    //Early return when this doesn't support our min api
                    //version
                    if props.properties.api_version < min_api_version.to_vk_version() {
                        return best_so_far;
                    }

                    let mut current_dev_mem_props = vk::PhysicalDeviceMemoryProperties2::default();

                    // Safety: current is derived from instance
                    unsafe {
                        instance.inner().get_physical_device_memory_properties2(
                            current,
                            &mut current_dev_mem_props,
                        );
                    }

                    // SAFETY: Surface and current are both derived from
                    // instance
                    let surface_capabilities = unsafe {
                        surface
                            .surface_instance()
                            .get_physical_device_surface_capabilities(current, surface.inner())
                    };

                    // Safety: surface and current are derived from instance
                    let formats = unsafe {
                        surface
                            .surface_instance()
                            .get_physical_device_surface_formats(current, surface.inner())
                    }
                    .ok()?;

                    // SAFETY: current and surface are derived from the same
                    // instance
                    let present_modes = unsafe {
                        surface
                            .surface_instance()
                            .get_physical_device_surface_present_modes(current, surface.inner())
                    };

                    let mut features_11 = vk::PhysicalDeviceVulkan11Features::default();
                    let mut features_12 = vk::PhysicalDeviceVulkan12Features::default();
                    let mut features_13 = vk::PhysicalDeviceVulkan13Features::default();
                    let mut features = vk::PhysicalDeviceFeatures2::default()
                        .push_next(&mut features_11)
                        .push_next(&mut features_12)
                        .push_next(&mut features_13);

                    // SAFETY: current is derived from instance
                    unsafe {
                        instance
                            .inner()
                            .get_physical_device_features2(current, &mut features);
                    }

                    let props_10 = props.properties;
                    let props_11 = props_11;
                    let props_12 = props_12;
                    let props_13 = props_13;

                    let features_10 = features.features;
                    let features_11 = features_11;
                    let features_12 = features_12;
                    let features_13 = features_13;
                    //SAFETY: Basically handled by ash doing the memory
                    //management for us
                    let qfis = unsafe {
                        instance
                            .inner()
                            .get_physical_device_queue_family_properties(current)
                    };
                    let graphics_qfi = qfis.iter().cloned().enumerate().find(|(qfi, props)| {
                        // SAFETY: current and surface come from the same
                        // instance
                        let present_support = unsafe {
                            surface
                                .surface_instance()
                                .get_physical_device_surface_support(
                                    current,
                                    *qfi as u32,
                                    surface.inner(),
                                )
                        }
                        .unwrap_or(false);
                        props.queue_flags.contains(QueueFlags::GRAPHICS) && present_support
                    });

                    let ext_names_iter = ext_props
                        .iter()
                        .map(|ext| ext.extension_name_as_c_str().unwrap_or(c""));

                    let swapchain_ext_present = ext_names_iter
                        .clone()
                        .any(|ext_name| ext_name.eq(ash::khr::swapchain::NAME));

                    //Check for mandatory features. This is currently a
                    //sample from vkguide.
                    if features_13.synchronization2 == vk::FALSE
                        || features_13.dynamic_rendering == vk::FALSE
                        || features_12.descriptor_indexing == vk::FALSE
                        || features_12.buffer_device_address == vk::FALSE
                        || !swapchain_ext_present
                        || graphics_qfi.is_none()
                    {
                        return best_so_far;
                    }
                    //SAFETY: We know that graphics_qfi.is_some is true. Established
                    //in the if statement for features
                    let graphics_qfi = unsafe { graphics_qfi.unwrap_unchecked() }.0 as u32;

                    //Preliminary scoring, not yet final
                    let device_type_score = match props_10.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                        vk::PhysicalDeviceType::OTHER | vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        vk::PhysicalDeviceType::CPU => 0,
                        _ => unreachable!(),
                    };
                    Some(ScoredPhysDev {
                        phys_dev: current,
                        score: device_type_score,
                        ext_props,
                        features_10,
                        features_11,
                        features_12,
                        features_13,
                        props_10,
                        props_11,
                        props_12,
                        props_13,
                        surface_capabilities,
                        formats,
                        present_modes,
                        graphics_qfi,
                    })
                }) {
                    Some(spd) => spd,
                    None => {
                        return Err(Error::NoSuitableDevice);
                    }
                };

            let enabled_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];

            let mut features_12 = PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .descriptor_indexing(true);

            let mut features_13 = PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);

            let mut features = PhysicalDeviceFeatures2::default()
                .push_next(&mut features_12)
                .push_next(&mut features_13);

            let qcis = [DeviceQueueCreateInfo::default()
                .queue_family_index(scored_phys_dev.graphics_qfi)
                .queue_priorities(&[1.0])];

            let dci = DeviceCreateInfo::default()
                .enabled_extension_names(&enabled_extensions)
                .push_next(&mut features)
                .queue_create_infos(&qcis);

            //SAFETY: Valid ci. phys_dev is derived from instance
            let device = unsafe {
                instance
                    .inner()
                    .create_device(scored_phys_dev.phys_dev, &dci, None)
            }?;

            let debug_utils_fps = instance
                .get_debug_messenger()
                .is_some()
                .then(|| ash::ext::debug_utils::Device::new(instance.inner(), &device));

            //SAFETY: graphics_qfi was passed in for one of the QCIs. Queue index is 0
            let graphics_queue =
                unsafe { device.get_device_queue(scored_phys_dev.graphics_qfi, 0) };

            Ok(Self {
                parent: instance.clone(),
                device,
                debug_utils_fps,
                phys_dev: scored_phys_dev.phys_dev,
                _graphics_queue: graphics_queue,
                graphics_queue_family_index: scored_phys_dev.graphics_qfi,
            })
        }

        /// Returns a handle to the associated physical device
        pub(crate) fn get_phys_dev(&self) -> vk::PhysicalDevice {
            self.phys_dev
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            self.debug_utils_fps.take();

            //SAFETY: Last use of device. All children are dead
            unsafe { self.device.destroy_device(None) };
        }
    }
}
/// All shader related functionality
pub mod shader {
    use std::{
        fmt::{Debug, Display},
        path::{Path, PathBuf},
        sync::Arc,
    };

    use ash::vk;
    use naga::{
        back::spv::{DebugInfo, Options, PipelineOptions, WriterFlags},
        front::glsl,
        valid::{Capabilities, SubgroupOperationSet, ValidationFlags},
    };
    use thiserror::Error;

    use super::device::Device;

    #[derive(Debug)]
    /// What type of shader we're talking about. Maybe we can exclude this if we
    /// have specific types for each shader? But right now this is good
    pub enum ShaderType {
        ///Vertex Shader (takes input from buffers and emits points)
        Vertex,
        ///Fragment Shader (What to render into the pixels)
        Fragment,
    }
    impl From<ShaderType> for naga::ShaderStage {
        fn from(val: ShaderType) -> Self {
            val.to_naga_stage()
        }
    }

    impl From<ShaderType> for naga::valid::ShaderStages {
        fn from(val: ShaderType) -> Self {
            val.to_naga_stage_bitfield()
        }
    }

    impl ShaderType {
        /// Conversion method
        fn to_naga_stage(&self) -> naga::ShaderStage {
            match self {
                ShaderType::Vertex => naga::ShaderStage::Vertex,
                ShaderType::Fragment => naga::ShaderStage::Fragment,
            }
        }
        /// Conversion method
        fn to_naga_stage_bitfield(&self) -> naga::valid::ShaderStages {
            match self {
                ShaderType::Vertex => naga::valid::ShaderStages::VERTEX,
                ShaderType::Fragment => naga::valid::ShaderStages::FRAGMENT,
            }
        }
    }
    #[derive(Debug)]
    /// Represents a shader object in vk with RAII semantics
    pub struct Shader {
        #[expect(dead_code, reason = "In dev")]
        /// What type this shader is
        shader_type: ShaderType,
        /// The underlying handle to the vulkan object
        vk_shader: vk::ShaderModule,
        /// Device we came from
        parent_device: Arc<Device>,
    }
    impl Shader {
        pub(crate) fn inner(&self) -> vk::ShaderModule {
            self.vk_shader
        }
    }

    #[derive(Debug, Error)]
    pub enum ShaderCompileError {
        /// Error parsing file
        Parse {
            /// The underlying naga error
            e: naga::front::glsl::ParseErrors,
            /// Source code
            src: String,
            /// Optional path to file
            file: Option<PathBuf>,
        },
        /// Error validating file
        Validation {
            /// Underlying naga error
            e: naga::WithSpan<naga::valid::ValidationError>,
            /// Source code
            src: String,
            /// Optional path to file
            path: Option<PathBuf>,
        },
        /// Some weird backend error
        Backend(naga::back::spv::Error, Option<PathBuf>),
        UnknownVulkan(vk::Result),
    }

    impl Display for ShaderCompileError {
        //TODO: Improve this Display impl. It sucks rn
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ShaderCompileError::Parse { e, src, file } => {
                    writeln!(
                        f,
                        "Could not create shader {} due to parse error.",
                        file.as_ref()
                            .and_then(|p| p.to_str())
                            .unwrap_or("<unknown>")
                    )?;
                    f.write_str(&e.emit_to_string(src))?;
                    Ok(())
                }
                ShaderCompileError::Validation { e, src, path } => f.write_str(&format!(
                    "Could not create shader {} due to validation error\n",
                    e.emit_to_string_with_path(
                        src,
                        path.as_ref()
                            .and_then(|p| p.to_str())
                            .unwrap_or("<unknown>")
                    )
                )),
                ShaderCompileError::Backend(error, path_buf) => {
                    f.write_str(&format!(
                        "Could not create shader {} due to backend error\n",
                        path_buf
                            .as_ref()
                            .and_then(|p| p.to_str())
                            .unwrap_or("<unknown>")
                    ))?;
                    Display::fmt(error, f)?;
                    Ok(())
                }
                ShaderCompileError::UnknownVulkan(e) => {
                    f.write_str("Could not create shader due to unknown vulkan error: ")?;
                    Display::fmt(&e, f)
                }
            }
        }
    }

    #[expect(dead_code, reason = "For now we only use one of these")]
    #[derive(Clone, Copy, Debug)]
    /// How much do we optimize these shaders
    pub enum ShaderOptLevel {
        /// Apply no optimizations
        None,
        /// Do actually optimize the shaders
        Heavy,
    }

    #[expect(dead_code, reason = "For now we only use one of these")]
    #[derive(Clone, Copy, Debug)]
    /// What level of debug info do we emit
    pub enum ShaderDebugLevel {
        /// Emit all debug info
        Full,
        /// Emit no debug info
        None,
    }

    /// Represents a compiler we can use to make shaders. Represented as an
    /// object to cache any relevant state
    pub struct ShaderCompiler {
        /// Responsible for taking the ir and emitting the final code
        spv_writer: naga::back::spv::Writer,
        /// How much shaders from this compiler should be optimized
        opt_level: ShaderOptLevel,
        /// What level of debug info to emit
        debug_level: ShaderDebugLevel,
        /// Parser for glsl
        glsl_parser: naga::front::glsl::Frontend,
        /// takes IR and ensures it is valid
        shader_validator: naga::valid::Validator,
        /// A device we can use to make the actual shader module. Technically
        /// not necessary but saves passing in different devices
        parent_device: Arc<Device>,
    }

    impl Debug for ShaderCompiler {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ShaderCompiler")
                .field("spv_writer", &"{..}")
                .field("opt_level", &self.opt_level)
                .field("debug_level", &self.debug_level)
                .finish()
        }
    }

    #[derive(Debug, Error)]
    /// Errors from compiling a shader
    pub enum ShaderCompilerError {
        #[error("Error creating naga backend: {0}")]
        /// Some weird backend error
        Backend(naga::back::spv::Error),
    }

    impl ShaderCompiler {
        /// Create a new shader compiler
        pub fn new(
            device: &Arc<Device>,
            debug_level: ShaderDebugLevel,
            opt_level: ShaderOptLevel,
        ) -> Result<Self, ShaderCompilerError> {
            let spv_writer = naga::back::spv::Writer::new(&Options {
                flags: match debug_level {
                    ShaderDebugLevel::Full => WriterFlags::DEBUG,
                    ShaderDebugLevel::None => WriterFlags::empty(),
                } | WriterFlags::ADJUST_COORDINATE_SPACE
                    | WriterFlags::CLAMP_FRAG_DEPTH
                    | WriterFlags::LABEL_VARYINGS,
                ..Default::default()
            })
            .map_err(ShaderCompilerError::Backend)?;

            let glsl_parser = naga::front::glsl::Frontend::default();

            let shader_validator =
                naga::valid::Validator::new(ValidationFlags::all(), Capabilities::all());

            Ok(Self {
                glsl_parser,
                shader_validator,
                spv_writer,
                opt_level,
                debug_level,
                parent_device: device.clone(),
            })
        }

        /// Create a shader object from its source and optionally its path
        pub fn compile_shader(
            &mut self,
            source: &str,
            shader_type: ShaderType,
            file_path: Option<&Path>,
        ) -> Result<Shader, Box<ShaderCompileError>> {
            let shader_mod = self
                .glsl_parser
                .parse(&glsl::Options::from(shader_type.to_naga_stage()), source)
                .map_err(|e| ShaderCompileError::Parse {
                    e,
                    src: source.to_owned(),
                    file: file_path.map(|p| p.to_owned()),
                })?;

            let shader_mod_info = self
                .shader_validator
                .subgroup_stages(shader_type.to_naga_stage_bitfield())
                .subgroup_operations(SubgroupOperationSet::all())
                .validate(&shader_mod)
                .map_err(|e| ShaderCompileError::Validation {
                    e,
                    src: source.to_owned(),
                    path: file_path.map(Path::to_owned),
                })?;

            //Ensure the shader validator is all good for next time
            self.shader_validator.reset();

            let mut spv_bytes = Vec::new();

            let debug_info = match (self.debug_level, file_path) {
                (ShaderDebugLevel::Full, Some(file_name)) => Some(DebugInfo {
                    source_code: source,
                    file_name,
                    language: naga::back::spv::SourceLanguage::GLSL,
                }),

                _ => None,
            };

            self.spv_writer
                .write(
                    &shader_mod,
                    &shader_mod_info,
                    Some(&PipelineOptions {
                        shader_stage: shader_type.to_naga_stage(),
                        entry_point: "main".to_owned(),
                    }),
                    &debug_info,
                    &mut spv_bytes,
                )
                .map_err(|e| ShaderCompileError::Backend(e, file_path.map(Path::to_owned)))?;

            let shader_mod_ci = vk::ShaderModuleCreateInfo::default().code(&spv_bytes);

            //SAFETY: We got spv_bytes from naga's backend so it *should* be valid Spir-V
            let vk_shader = unsafe {
                self.parent_device
                    .inner()
                    .create_shader_module(&shader_mod_ci, None)
            }
            .map_err(ShaderCompileError::UnknownVulkan)?;

            Ok(Shader {
                vk_shader,
                shader_type,
                parent_device: self.parent_device.clone(),
            })
        }
    }

    impl Drop for Shader {
        fn drop(&mut self) {
            //SAFETY: We know that self.vk_shader was made from
            //self.parent_device
            unsafe {
                self.parent_device
                    .inner()
                    .destroy_shader_module(self.vk_shader, None)
            };
        }
    }
}
