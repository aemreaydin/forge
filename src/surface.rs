use crate::{instance, vulkan_handle};
use ash::{khr, vk};

pub struct Surface {
    surface: vk::SurfaceKHR,
    loader: khr::surface::Instance,
}

vulkan_handle!(Surface, surface, vk::SurfaceKHR);

impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: &instance::Instance,
        window: &sdl3::video::Window,
    ) -> anyhow::Result<Self> {
        let loader = khr::surface::Instance::new(entry, instance.handle());
        let surface = window.vulkan_create_surface(instance.handle().handle())?;

        Ok(Self { surface, loader })
    }

    pub fn destroy(&self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }

    pub fn get_physical_device_surface_support_khr(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> anyhow::Result<bool> {
        unsafe {
            Ok(self.loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index,
                self.surface,
            )?)
        }
    }
    pub fn get_physical_device_surface_capabilities_khr(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            Ok(self
                .loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)?)
        }
    }

    pub fn get_physical_device_surface_formats_khr(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            Ok(self
                .loader
                .get_physical_device_surface_formats(physical_device, self.surface)?)
        }
    }

    pub fn get_physical_device_surface_present_modes_khr(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Vec<vk::PresentModeKHR>> {
        unsafe {
            Ok(self
                .loader
                .get_physical_device_surface_present_modes(physical_device, self.surface)?)
        }
    }
}
