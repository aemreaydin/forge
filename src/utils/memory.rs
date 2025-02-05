use crate::context::VulkanContext;
use anyhow::Context;

pub fn get_required_memory_index(
    vulkan_context: &VulkanContext,
    memory_requirements: ash::vk::MemoryRequirements,
    required_memory_flags: ash::vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    Ok(vulkan_context
        .physical_device
        .memory_properties
        .memory_types
        .iter()
        .enumerate()
        .position(|(ind, mem_type)| {
            mem_type.property_flags.contains(required_memory_flags)
                && (memory_requirements.memory_type_bits & (1 << ind)) != 0
        })
        .context("failed to find a suitable memory type index")? as u32)
}
