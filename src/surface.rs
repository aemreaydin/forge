use crate::instance::Instance;
use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub instance: Arc<Instance>,
    loader: khr::surface::Instance,
}

impl Surface {
    pub fn new(
        instance: Arc<Instance>,
        handle: &(impl HasDisplayHandle + HasWindowHandle),
    ) -> anyhow::Result<Self> {
        unsafe {
            let entry = &instance.entry;
            let loader = khr::surface::Instance::new(entry, &instance.instance);

            let raw_display_handle = handle
                .display_handle()
                .expect("failed to get raw display handle")
                .as_raw();
            let raw_window_handle = handle
                .window_handle()
                .expect("failed to get raw window handle")
                .as_raw();

            let surface = ash_window::create_surface(
                entry,
                &instance.instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?;

            Ok(Self {
                surface,
                loader,
                instance,
            })
        }
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
