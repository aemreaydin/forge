use super::{device, instance, physical_device, surface, swapchain};
use ash::vk;
use std::sync::Arc;

#[derive(Clone)]
pub struct VulkanContext {
    pub instance: Arc<instance::Instance>,
    pub device: Arc<device::Device>,

    swapchain: swapchain::Swapchain,
    pub physical_device: physical_device::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
}

impl VulkanContext {
    pub fn new(
        entry: ash::Entry,
        instance: instance::Instance,
        surface: vk::SurfaceKHR,
    ) -> anyhow::Result<Self> {
        let mut surface = surface::Surface::new(&entry, &instance, surface)?;
        let physical_device = physical_device::PhysicalDevice::new(&instance, &surface)?;
        surface.set_format(physical_device.physical_device)?;

        let device = device::Device::new(&instance.instance, &physical_device)?;
        let swapchain = swapchain::Swapchain::new(
            &instance.instance,
            physical_device.physical_device,
            &device.device,
            surface,
        )?;

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
            device: Arc::new(device),
            swapchain,
            physical_device,
            graphics_queue,
            compute_queue,
            transfer_queue,
        })
    }

    pub fn resized(&mut self) -> anyhow::Result<()> {
        self.swapchain
            .recreate(self.physical_device.physical_device, &self.device.device)?;
        Ok(())
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance.instance
    }

    pub fn device(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device.physical_device
    }

    pub fn swapchain(&self) -> &swapchain::Swapchain {
        &self.swapchain
    }

    pub fn surface_format(&self) -> vk::SurfaceFormatKHR {
        self.swapchain.surface.format
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        self.swapchain.destroy(self.device());
        self.device.destroy();
        self.instance.destroy();
    }
}
