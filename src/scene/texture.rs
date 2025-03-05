use crate::{image::Image, vulkan_context::VulkanContext};
use ash::vk;

#[derive(Debug)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Texture {
    pub texture_data: TextureData,
    pub image: Image,
}

impl Texture {
    pub fn from_2d_data(
        vulkan_context: &VulkanContext,
        texture_data: TextureData,
    ) -> anyhow::Result<Self> {
        let extent = vk::Extent3D {
            width: texture_data.width,
            height: texture_data.height,
            depth: 1,
        };
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(extent)
            .mip_levels(1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = Image::from_data(
            &vulkan_context.physical_device,
            &vulkan_context.device,
            &texture_data.data,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            image_create_info,
            vk::ImageViewType::TYPE_2D,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1),
        )?;

        Ok(Self {
            texture_data,
            image,
        })
    }

    pub fn destroy(&self, vulkan_context: &VulkanContext) {
        self.image.destroy(vulkan_context.device());
    }
}
