use crate::{
    buffer::{Buffer, Vertex},
    vulkan_context::VulkanContext,
};
use ash::vk;

#[derive(Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl Mesh {
    pub fn new(
        vulkan_context: &VulkanContext,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> anyhow::Result<Self> {
        // TODO: Check use of staging buffers here
        let vertex_buffer = Buffer::from_data(
            &vulkan_context.physical_device,
            vulkan_context.device(),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vertices,
        )?;
        let index_buffer = Buffer::from_data(
            &vulkan_context.physical_device,
            vulkan_context.device(),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        )?;

        Ok(Self {
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
        })
    }

    pub fn destroy(&self, vulkan_context: &VulkanContext) {
        self.vertex_buffer.destroy(vulkan_context.device());
        self.index_buffer.destroy(vulkan_context.device());
    }
}
