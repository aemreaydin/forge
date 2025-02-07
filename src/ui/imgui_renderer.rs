use super::imgui_sdl3_platform::ImguiSdlPlatform;
use crate::renderer::{buffer::Buffer, vulkan_context::VulkanContext, Renderer};
use ash::vk;
use imgui::Context;
use sdl3::event::Event;
use std::sync::Arc;

pub struct ImguiVulkanRenderer {
    vulkan_context: VulkanContext,
    sdl_platform: ImguiSdlPlatform,
    imgui_context: Context,

    prev_idx_count: usize,
    prev_vtx_count: usize,

    vtx_buffer: Option<Buffer>,
    idx_buffer: Option<Buffer>,
    // Resources
    //render_pass: vk::RenderPass,
    //pipeline_layout: vk::PipelineLayout,
    //pipeline: vk::Pipeline,
    //framebuffers: Vec<vk::Framebuffer>,
    //descriptor_set_layout: vk::DescriptorSetLayout,
    //font_texture: vk::DescriptorSet,
    //font_sampler: vk::Sampler,
}

impl ImguiVulkanRenderer {
    pub fn new(vulkan_context: VulkanContext, sdl_platform: ImguiSdlPlatform) -> Self {
        let imgui_context = Context::create();
        Self {
            vulkan_context,
            sdl_platform,
            imgui_context,

            prev_idx_count: 0,
            prev_vtx_count: 0,

            vtx_buffer: None,
            idx_buffer: None,
        }
    }

    pub fn process_event(&mut self, event: &Event) {
        self.sdl_platform
            .process_event(&mut self.imgui_context, event);
    }

    fn update_buffers(&mut self) -> Buffer {
        todo!()
    }
}

impl Renderer for ImguiVulkanRenderer {
    fn start_frame(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn end_frame(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn draw(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn resized(&mut self, dims: &[u32; 2]) -> anyhow::Result<bool> {
        todo!()
    }
}
