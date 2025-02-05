use super::{device, instance, physical_device, surface};
use std::sync::Arc;

pub struct VulkanContext {
    instance: Arc<instance::Instance>,
    pub surface: Arc<surface::Surface>,
    pub device: Arc<device::Device>,

    pub physical_device: physical_device::PhysicalDevice,
    pub graphics_queue: ash::vk::Queue,
    pub compute_queue: ash::vk::Queue,
    pub transfer_queue: ash::vk::Queue,
}

impl VulkanContext {
    pub fn new(
        entry: ash::Entry,
        instance: instance::Instance,
        surface: ash::vk::SurfaceKHR,
    ) -> anyhow::Result<Self> {
        let surface = surface::Surface::new(&entry, &instance, surface)?;

        let physical_device = physical_device::PhysicalDevice::new(&instance, &surface)?;
        let device = device::Device::new(&instance.instance, &physical_device)?;

        let graphics_queue = unsafe {
            device
                .device
                .get_device_queue(physical_device.queue_indices.graphics, 0)
        };
        let compute_queue = unsafe {
            device
                .device
                .get_device_queue(physical_device.queue_indices.compute, 0)
        };
        let transfer_queue = unsafe {
            device
                .device
                .get_device_queue(physical_device.queue_indices.transfer, 0)
        };

        Ok(Self {
            instance: Arc::new(instance),
            surface: Arc::new(surface),
            device: Arc::new(device),
            physical_device,
            graphics_queue,
            compute_queue,
            transfer_queue,
        })
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance.instance
    }

    pub fn device(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn physical_device(&self) -> ash::vk::PhysicalDevice {
        self.physical_device.physical_device
    }

    pub fn surface(&self) -> ash::vk::SurfaceKHR {
        self.surface.surface
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.device.destroy();
        self.surface.destroy();
        self.instance.destroy();
    }
}
