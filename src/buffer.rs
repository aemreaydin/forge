use anyhow::Context;
use ash::{vk, Device};
use std::ptr::copy;

pub trait DeviceHandled {
    fn device(&self) -> &Device;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

pub struct Buffer<T> {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub data: Vec<T>,

    device: Device,
}

impl<T> Buffer<T> {
    pub fn from_data(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        required_memory_flags: vk::MemoryPropertyFlags,
        usage: vk::BufferUsageFlags,
        data: Vec<T>,
    ) -> anyhow::Result<Self> {
        let size = (size_of::<T>() * data.len()).try_into()?;

        let create_info = vk::BufferCreateInfo::default().size(size).usage(usage);

        unsafe {
            let buffer = device.create_buffer(&create_info, None)?;

            let mem_req = device.get_buffer_memory_requirements(buffer);
            let mem_index = memory_properties
                .memory_types
                .iter()
                .enumerate()
                .position(|(ind, mem_type)| {
                    mem_type.property_flags.contains(required_memory_flags)
                        && (mem_req.memory_type_bits & (1 << ind)) != 0
                })
                .context("failed to find a suitable memory type index")?;
            log::trace!("Picking memory index {}.", mem_index);

            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_index as u32);
            let memory = device.allocate_memory(&allocate_info, None)?;

            device.bind_buffer_memory(buffer, memory, 0)?;
            let mapped_memory = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;

            copy(data.as_ptr(), mapped_memory.cast(), data.len());

            Ok(Self {
                buffer,
                memory,
                data,
                device: device.clone(),
            })
        }
    }
}

impl<T> DeviceHandled for Buffer<T> {
    fn device(&self) -> &Device {
        &self.device
    }
}
