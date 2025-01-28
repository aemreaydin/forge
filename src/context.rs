use crate::{device, instance, physical_device, surface, utils::handle::Handle};
use std::sync::Arc;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub struct VulkanContext {
    instance: Arc<instance::Instance>,

    pub surface: surface::Surface,
    pub physical_device: physical_device::PhysicalDevice,
    pub device: device::Device,

    pub graphics_queue: ash::vk::Queue,
    pub compute_queue: ash::vk::Queue,
    pub transfer_queue: ash::vk::Queue,
}

impl VulkanContext {
    pub fn new(window: &sdl3::video::Window) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = instance::Instance::new(&entry, VALIDATION_ENABLED)?;

        let surface = surface::Surface::new(&entry, &instance, window)?;

        let physical_device = physical_device::PhysicalDevice::new(&instance, &surface)?;
        let device = device::Device::new(&instance, &physical_device)?;

        let graphics_queue = unsafe {
            device
                .handle()
                .get_device_queue(physical_device.queue_indices.graphics, 0)
        };
        let compute_queue = unsafe {
            device
                .handle()
                .get_device_queue(physical_device.queue_indices.compute, 0)
        };
        let transfer_queue = unsafe {
            device
                .handle()
                .get_device_queue(physical_device.queue_indices.transfer, 0)
        };

        Ok(Self {
            instance,

            surface,

            physical_device,
            device,

            graphics_queue,
            compute_queue,
            transfer_queue,
        })
    }

    pub fn instance(&self) -> &ash::Instance {
        self.instance.handle()
    }

    pub fn device(&self) -> &ash::Device {
        self.device.handle()
    }

    pub fn physical_device(&self) -> ash::vk::PhysicalDevice {
        *self.physical_device.handle()
    }

    pub fn surface(&self) -> ash::vk::SurfaceKHR {
        *self.surface.handle()
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.device.destroy();
        self.surface.destroy();
        self.instance.destroy();
    }
}
