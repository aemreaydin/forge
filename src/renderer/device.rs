use super::physical_device::PhysicalDevice;
use anyhow::anyhow;
use ash::vk;
use std::ffi::CStr;

const REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[
    vk::KHR_SWAPCHAIN_NAME.as_ptr(),
    vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr(),
];

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_DEVICE_EXTENSIONS: &[*const i8] =
    &[ash::khr::portability_subset::NAME.as_ptr()];

const OPTIONAL_DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::dynamic_rendering::NAME,
    ash::ext::shader_object::NAME,
];

#[derive(Debug, Default)]
pub struct DeviceSupport {
    pub shader_ext: bool,
    pub dynamic_rendering: bool,
}

pub struct Device {
    pub device: ash::Device,
    pub device_support: DeviceSupport,

    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub transfer_queue: vk::Queue,

    pub graphics_command_pool: vk::CommandPool,
}

impl Device {
    pub fn new(instance: &ash::Instance, physical_device: &PhysicalDevice) -> anyhow::Result<Self> {
        unsafe {
            let device_extensions =
                instance.enumerate_device_extension_properties(physical_device.physical_device)?;
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

            let mut push_descriptor_props =
                vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut physical_device_props =
                vk::PhysicalDeviceProperties2::default().push_next(&mut push_descriptor_props);
            instance.get_physical_device_properties2(
                physical_device.physical_device,
                &mut physical_device_props,
            );

            let mut shader_object =
                vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
            let mut dynamic_rendering =
                vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default().dynamic_rendering(true);
            let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
                .features(vk::PhysicalDeviceFeatures::default())
                .push_next(&mut dynamic_rendering)
                .push_next(&mut shader_object);

            let device_support = DeviceSupport::default();
            // optional_extensions.iter().for_each(|&ext| match ext {
            //     val if { CStr::from_ptr(val) } == ext::shader_object::NAME => {
            //         device_support.shader_ext = shader_object.shader_object != 0;
            //     }
            //     val if { CStr::from_ptr(val) } == ash::khr::dynamic_rendering::NAME => {
            //         log::info!("Supporting dynamic_rendering");
            //         device_support.dynamic_rendering = dynamic_rendering.dynamic_rendering != 0;
            //     }
            //     _ => {}
            // });

            let create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&extensions)
                .push_next(&mut physical_device_features);
            let device =
                instance.create_device(physical_device.physical_device, &create_info, None)?;

            let graphics_queue = device.get_device_queue(physical_device.queue_indices.graphics, 0);
            let compute_queue = device.get_device_queue(physical_device.queue_indices.compute, 0);
            let transfer_queue = device.get_device_queue(physical_device.queue_indices.transfer, 0);

            let graphics_command_pool = crate::create_command_pool(
                &device,
                physical_device.queue_indices.graphics,
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            Ok(Self {
                device,
                device_support,

                graphics_queue,
                compute_queue,
                transfer_queue,

                graphics_command_pool,
            })
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
            self.device.destroy_device(None);
        }
    }

    fn get_required_device_extensions(
        device_extensions: &[vk::ExtensionProperties],
    ) -> anyhow::Result<Vec<*const i8>> {
        let mut required_extensions = REQUIRED_DEVICE_EXTENSIONS.to_vec();
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
        let device_extension_names = device_extensions
            .iter()
            .map(|extension| extension.extension_name_as_c_str().unwrap())
            .collect::<Vec<_>>();

        OPTIONAL_DEVICE_EXTENSIONS
            .iter()
            .filter_map(|&ext| {
                if device_extension_names.contains(&ext) {
                    log::info!("{} is supported", ext.to_string_lossy());
                    Some(ext.as_ptr())
                } else {
                    log::warn!(
                        "{} is not supported by current device",
                        ext.to_string_lossy()
                    );
                    None
                }
            })
            .collect::<Vec<_>>()
    }
}
