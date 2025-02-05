use super::physical_device::PhysicalDevice;
use anyhow::anyhow;
use ash::{ext, khr, vk};
use std::ffi::CStr;

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[khr::portability_subset::NAME.as_ptr()];

const OPTIONAL_DEVICE_EXTENSIONS: &[*const i8] = &[
    ext::shader_object::NAME.as_ptr(),
    khr::dynamic_rendering::NAME.as_ptr(),
];

#[derive(Debug, Default)]
pub struct DeviceSupport {
    pub shader_ext: bool,
    pub dynamic_rendering: bool,
}

pub struct Device {
    pub device: ash::Device,
    pub device_support: DeviceSupport,
}

impl Device {
    pub fn new(instance: &ash::Instance, physical_device: &PhysicalDevice) -> anyhow::Result<Self> {
        let device_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device.physical_device)?
        };
        let required_extensions = Self::get_required_device_extensions(&device_extensions)?;
        let optional_extensions = Self::get_optional_device_extensions(&device_extensions);
        let extensions = required_extensions
            .iter()
            .chain(optional_extensions.iter())
            .cloned()
            .collect::<Vec<_>>();
        let queue_create_infos = [
            vk::DeviceQueueCreateInfo {
                queue_count: 1,
                queue_family_index: physical_device.queue_indices.graphics,
                ..Default::default()
            }
            .queue_priorities(&[1.0]),
            vk::DeviceQueueCreateInfo {
                queue_count: 1,
                queue_family_index: physical_device.queue_indices.compute,
                ..Default::default()
            }
            .queue_priorities(&[1.0]),
            vk::DeviceQueueCreateInfo {
                queue_count: 1,
                queue_family_index: physical_device.queue_indices.transfer,
                ..Default::default()
            }
            .queue_priorities(&[1.0]),
        ];

        // TODO: Probably clean this up
        let physical_device_features = vk::PhysicalDeviceFeatures::default().depth_bounds(true);
        let mut shader_object =
            vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
        let mut dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default().dynamic_rendering(true);
        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .features(physical_device_features)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut shader_object);

        let mut device_support = DeviceSupport::default();
        optional_extensions.iter().for_each(|&ext| match ext {
            val if unsafe { CStr::from_ptr(val) } == ext::shader_object::NAME => {
                device_support.shader_ext = false;
            }
            val if unsafe { CStr::from_ptr(val) } == khr::dynamic_rendering::NAME => {
                device_support.dynamic_rendering = false;
            }
            _ => {}
        });

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .push_next(&mut physical_device_features);
        let device =
            unsafe { instance.create_device(physical_device.physical_device, &create_info, None)? };

        Ok(Self {
            device,
            device_support,
        })
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }

    fn get_required_device_extensions(
        device_extensions: &[vk::ExtensionProperties],
    ) -> anyhow::Result<Vec<*const i8>> {
        #[allow(unused_mut)]
        let mut required_extensions = vec![
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr(),
        ];
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        required_extensions.extend_from_slice(MACOS_REQUIRED_DEVICE_EXTENSIONS);

        let device_extension_names = unsafe {
            device_extensions
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

    fn get_optional_device_extensions(
        device_extensions: &[vk::ExtensionProperties],
    ) -> Vec<*const i8> {
        #[allow(unused_mut)]
        let device_extension_names = unsafe {
            device_extensions
                .iter()
                .map(|extension| std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()))
                .collect::<Vec<_>>()
        };

        OPTIONAL_DEVICE_EXTENSIONS
            .iter()
            .filter_map(|ext| {
                let name = unsafe { std::ffi::CStr::from_ptr(*ext) };
                if device_extension_names.contains(&name) {
                    Some(*ext)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }
}
