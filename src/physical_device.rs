use crate::{instance::Instance, surface::Surface};
use anyhow::anyhow;
use ash::vk::{self, PhysicalDeviceMemoryProperties};

fn i8_array_to_string(slice: &[i8]) -> String {
    String::from_utf8_lossy(
        &slice
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect::<Vec<u8>>(),
    )
    .to_string()
}

pub struct PhysicalDevice {
    pub handle: ash::vk::PhysicalDevice,

    pub memory_properties: PhysicalDeviceMemoryProperties,
}

impl PhysicalDevice {
    pub fn new(instance: &Instance, surface: &Surface) -> anyhow::Result<Self> {
        let physical_devices = unsafe { instance.instance.enumerate_physical_devices()? };

        let mut selected_device: Option<vk::PhysicalDevice> = None;
        let mut fallback_device: Option<vk::PhysicalDevice> = None;
        for device in &physical_devices {
            let properties = unsafe { instance.instance.get_physical_device_properties(*device) };
            let surface_support = surface.get_physical_device_surface_support_khr(*device, 0)?; // TODO: Queue family index is hardcoded
                                                                                                // TODO: These are not being used right now
            let _features = unsafe { instance.instance.get_physical_device_features(*device) };
            let is_discrete = properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

            match surface_support {
                true if is_discrete => {
                    selected_device = Some(*device);
                }
                true => fallback_device = Some(*device),
                false => {}
            }
        }

        match (selected_device, fallback_device) {
            (Some(device), _) => {
                let props = unsafe { instance.instance.get_physical_device_properties(device) };
                log::info!(
                    "Using discrete gpu: {}",
                    i8_array_to_string(&props.device_name)
                );
                Ok(Self {
                    handle: device,
                    memory_properties: Self::get_memory_properties(instance, device),
                })
            }
            (_, Some(fallback)) => {
                let props = unsafe { instance.instance.get_physical_device_properties(fallback) };
                log::info!(
                    "Using fallback device: {}",
                    i8_array_to_string(&props.device_name)
                );
                Ok(Self {
                    handle: fallback,
                    memory_properties: Self::get_memory_properties(instance, fallback),
                })
            }
            _ => Err(anyhow!("no suitable physical devices found")),
        }
    }

    fn get_memory_properties(
        instance: &Instance,
        device: vk::PhysicalDevice,
    ) -> PhysicalDeviceMemoryProperties {
        unsafe {
            instance
                .instance
                .get_physical_device_memory_properties(device)
        }
    }
}
