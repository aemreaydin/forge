use std::sync::OnceLock;

use super::instance;
use anyhow::Context;
use ash::{khr, vk};

static SURFACE_INSTANCE_FNS: OnceLock<khr::surface::Instance> = OnceLock::new();

pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
}

impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: &instance::Instance,
        surface: vk::SurfaceKHR,
    ) -> anyhow::Result<Self> {
        SURFACE_INSTANCE_FNS.get_or_init(|| khr::surface::Instance::new(entry, &instance.instance));
        Ok(Self {
            surface,
            format: vk::SurfaceFormatKHR::default(),
        })
    }

    pub fn destroy(&self) {
        unsafe {
            Self::get_instance_fns().destroy_surface(self.surface, None);
        }
    }

    pub fn set_format(&mut self, physical_device: vk::PhysicalDevice) -> anyhow::Result<()> {
        let surface_formats = self.get_physical_device_surface_formats_khr(physical_device)?;
        // TODO: This could be more sophisticated
        self.format = surface_formats
            .into_iter()
            .find(|format| {
                !(format.color_space != vk::ColorSpaceKHR::SRGB_NONLINEAR
                    || format.format != vk::Format::R8G8B8A8_SRGB
                        && format.format != vk::Format::B8G8R8A8_SRGB)
            })
            .context("failed to find a suitable surface format")?;
        Ok(())
    }

    pub fn get_physical_device_surface_support_khr(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> anyhow::Result<bool> {
        unsafe {
            Ok(
                Self::get_instance_fns().get_physical_device_surface_support(
                    physical_device,
                    queue_family_index,
                    self.surface,
                )?,
            )
        }
    }

    pub fn get_physical_device_surface_capabilities_khr(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<vk::SurfaceCapabilitiesKHR> {
        unsafe {
            Ok(Self::get_instance_fns()
                .get_physical_device_surface_capabilities(physical_device, self.surface)?)
        }
    }

    pub fn get_physical_device_surface_formats_khr(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            Ok(Self::get_instance_fns()
                .get_physical_device_surface_formats(physical_device, self.surface)?)
        }
    }

    pub fn get_physical_device_surface_present_modes_khr(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> anyhow::Result<Vec<vk::PresentModeKHR>> {
        unsafe {
            Ok(Self::get_instance_fns()
                .get_physical_device_surface_present_modes(physical_device, self.surface)?)
        }
    }

    fn get_instance_fns() -> &'static khr::surface::Instance {
        SURFACE_INSTANCE_FNS
            .get()
            .expect("Surface instance fns should be initialized by now.")
    }
}
