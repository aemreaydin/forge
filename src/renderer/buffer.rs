use super::physical_device::PhysicalDevice;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::{ffi::c_void, ptr::copy};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: nalgebra_glm::Vec4,
    pub normal: nalgebra_glm::Vec4,
    pub tex_coords: nalgebra_glm::Vec2,
    pub _padding: nalgebra_glm::Vec2, // TODO: There has to be a way to do this without padding
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: nalgebra_glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
            normal: nalgebra_glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
            tex_coords: nalgebra_glm::Vec2::new(0.0, 0.0),
            _padding: nalgebra_glm::Vec2::new(0.0, 0.0),
        }
    }
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl Buffer {
    pub fn new(
        physical_device: &PhysicalDevice,
        device: &ash::Device,
        size: vk::DeviceSize,
        required_memory_flags: vk::MemoryPropertyFlags,
        usage: vk::BufferUsageFlags,
    ) -> anyhow::Result<Self> {
        let create_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        unsafe {
            let buffer = device.create_buffer(&create_info, None)?;
            let mem_req = device.get_buffer_memory_requirements(buffer);
            let mem_index =
                physical_device.get_required_memory_index(mem_req, required_memory_flags)?;
            log::trace!("Picking memory index {} for buffer.", mem_index);

            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_index);
            let memory = device.allocate_memory(&allocate_info, None)?;

            device.bind_buffer_memory(buffer, memory, 0)?;

            Ok(Self {
                buffer,
                memory,
                size,
            })
        }
    }

    pub fn from_data<T: Pod>(
        physical_device: &PhysicalDevice,
        device: &ash::Device,
        required_memory_flags: vk::MemoryPropertyFlags,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> anyhow::Result<Self> {
        let size = std::mem::size_of_val(data).try_into()?;
        let buffer = Self::new(physical_device, device, size, required_memory_flags, usage)?;
        let mapped_memory = buffer.map_memory(device)?;
        unsafe {
            copy(data.as_ptr(), mapped_memory.cast(), data.len());
        }
        Ok(buffer)
    }

    pub fn map_memory(&self, device: &ash::Device) -> anyhow::Result<*mut c_void> {
        unsafe { Ok(device.map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?) }
    }

    pub fn unmap_memory(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory);
        }
    }

    pub fn flush_mapped_memory_ranges(
        &self,
        device: &ash::Device,
        mapped_memory_ranges: &[vk::MappedMemoryRange],
    ) -> anyhow::Result<()> {
        unsafe {
            device.flush_mapped_memory_ranges(mapped_memory_ranges)?;
            self.unmap_memory(device);
        }

        Ok(())
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("buffer", &self.buffer)
            .field("memory", &self.memory)
            .field("size", &self.size)
            .finish()
    }
}
