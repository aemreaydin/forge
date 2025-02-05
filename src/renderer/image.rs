use super::{buffer::Buffer, context::VulkanContext};
use anyhow::Context;
use ash::vk;
use bytemuck::Pod;

pub struct Image {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub size: u64,

    extent: vk::Extent3D,
}

impl Image {
    pub fn new(
        vulkan_context: &VulkanContext,
        size: u64,
        required_memory_flags: vk::MemoryPropertyFlags,
        image_create_info: vk::ImageCreateInfo,
        view_type: vk::ImageViewType,
        subresource_range: vk::ImageSubresourceRange,
    ) -> anyhow::Result<Self> {
        unsafe {
            let image = vulkan_context
                .device()
                .create_image(&image_create_info, None)?;

            let mem_req = vulkan_context.device().get_image_memory_requirements(image);
            let mem_index = vulkan_context
                .physical_device
                .memory_properties
                .memory_types
                .iter()
                .enumerate()
                .position(|(ind, mem_type)| {
                    mem_type.property_flags.contains(required_memory_flags)
                        && (mem_req.memory_type_bits & (1 << ind)) != 0
                })
                .context("failed to find a suitable memory type index")?;
            log::trace!("Picking memory index {} for image.", mem_index);

            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_index as u32);
            let memory = vulkan_context
                .device()
                .allocate_memory(&allocate_info, None)?;

            vulkan_context
                .device()
                .bind_image_memory(image, memory, 0)?;

            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .format(image_create_info.format)
                .view_type(view_type)
                .subresource_range(subresource_range);
            let image_view = vulkan_context
                .device()
                .create_image_view(&image_view_create_info, None)?;

            Ok(Self {
                image,
                image_view,
                memory,
                size,

                extent: image_create_info.extent,
            })
        }
    }

    pub fn copy_to_host<T: Pod>(
        &self,
        vulkan_context: &VulkanContext,
        cmd: vk::CommandBuffer,
        data: &[T],
    ) -> anyhow::Result<()> {
        unsafe {
            let upload_buffer = Buffer::from_data(
                vulkan_context,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::BufferUsageFlags::TRANSFER_SRC,
                data,
            )?;
            let mapped_memory_ranges = &[vk::MappedMemoryRange::default()
                .memory(upload_buffer.memory)
                .size(upload_buffer.size)];
            upload_buffer
                .flush_mapped_memory_ranges(vulkan_context.device(), mapped_memory_ranges)?;

            let copy_barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            vulkan_context.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[copy_barrier],
            );

            let buffer_image_copy = &[vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(self.extent)];
            vulkan_context.device().cmd_copy_buffer_to_image(
                cmd,
                upload_buffer.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                buffer_image_copy,
            );

            let use_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            vulkan_context.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[use_barrier],
            );

            let cmds = &[cmd];
            let font_submit_info = vk::SubmitInfo::default().command_buffers(cmds);
            vulkan_context.device().end_command_buffer(cmd)?;
            vulkan_context.device().queue_submit(
                vulkan_context.graphics_queue,
                &[font_submit_info],
                vk::Fence::null(),
            )?;
            vulkan_context
                .device()
                .queue_wait_idle(vulkan_context.graphics_queue)?;

            upload_buffer.destroy(vulkan_context.device());
        }

        Ok(())
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image(self.image, None);
            device.destroy_image_view(self.image_view, None);
            device.free_memory(self.memory, None);
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("image", &self.image)
            .field("memory", &self.memory)
            .field("size", &self.size)
            .finish()
    }
}
