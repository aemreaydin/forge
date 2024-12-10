use crate::instance::Instance;
use anyhow::anyhow;
use ash::{ext, vk};
use std::ffi::CStr;

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[khr::portability_subset::NAME.as_ptr()];

const OPTIONAL_DEVICE_EXTENSIONS: &[&CStr] = &[ext::shader_object::NAME];

#[derive(Debug, Default)]
pub struct DeviceSupport {
    pub shader_ext: bool,
}

pub struct Device {
    pub handle: ash::Device,
    pub device_support: DeviceSupport,
}

impl Device {
    pub fn new(instance: &Instance, physical_device: &vk::PhysicalDevice) -> anyhow::Result<Self> {
        let extensions = Self::get_required_device_extensions(instance, physical_device)?;
        let queue_create_infos = [vk::DeviceQueueCreateInfo {
            queue_count: 1,
            queue_family_index: 0,
            ..Default::default()
        }
        .queue_priorities(&[1.0])];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions);
        let handle = unsafe {
            instance
                .instance
                .create_device(*physical_device, &create_info, None)?
        };

        let mut device_support = DeviceSupport { shader_ext: false };
        OPTIONAL_DEVICE_EXTENSIONS
            .iter()
            .for_each(|&ext| match ext {
                val if val == ext::shader_object::NAME => {
                    device_support.shader_ext = true;
                }
                _ => {}
            });

        Ok(Self {
            handle,
            device_support,
        })
    }

    fn get_required_device_extensions(
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> anyhow::Result<Vec<*const i8>> {
        #[allow(unused_mut)]
        let mut required_extensions = vec![
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr(),
        ];
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        required_extensions.extend_from_slice(MACOS_REQUIRED_DEVICE_EXTENSIONS);

        let device_extension_names = unsafe {
            instance
                .instance
                .enumerate_device_extension_properties(*physical_device)?
                .iter()
                .map(|extension| std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()))
                .collect::<Vec<_>>()
        };
        for required_extension in &required_extensions {
            let extension_name = unsafe { std::ffi::CStr::from_ptr(*required_extension) };
            if !device_extension_names.contains(&extension_name) {
                return Err(anyhow!(
                    "extension {} not supported by the device",
                    extension_name.to_string_lossy()
                ));
            }
        }
        Ok(required_extensions)
    }
}
