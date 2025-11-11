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

impl DynamicPipeline {
    /// Get the raw Vulkan pipeline handle.
    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }
}
