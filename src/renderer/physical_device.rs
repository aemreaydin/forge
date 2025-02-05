use super::{instance::Instance, surface::Surface};
use anyhow::{anyhow, Context};
use ash::vk;

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

#[derive(Debug)]
pub struct QueueIndices {
    pub graphics: u32,
    pub compute: u32,
    pub transfer: u32,
}

pub struct PhysicalDevice {
    pub physical_device: vk::PhysicalDevice,

    pub queue_indices: QueueIndices,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl PhysicalDevice {
    pub fn new(instance: &Instance, surface: &Surface) -> anyhow::Result<Self> {
        let physical_devices = unsafe { instance.instance.enumerate_physical_devices()? };

        let mut selected_device: Option<vk::PhysicalDevice> = None;
        let mut fallback_device: Option<vk::PhysicalDevice> = None;
        let mut queue_indices = None;
        for device in &physical_devices {
            let properties = unsafe { instance.instance.get_physical_device_properties(*device) };
            let is_discrete = properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

            queue_indices = Some(Self::find_queue_family_indices(instance, surface, device)?);

            if is_discrete {
                selected_device = Some(*device);
            } else {
                fallback_device = Some(*device);
            }
        }

        let physical_device = match (selected_device, fallback_device) {
            (Some(device), _) => {
                let props = unsafe { instance.instance.get_physical_device_properties(device) };
                log::info!(
                    "Using discrete gpu: {}",
                    i8_array_to_string(&props.device_name)
                );
                Ok(device)
            }
            (_, Some(fallback)) => {
                let props = unsafe { instance.instance.get_physical_device_properties(fallback) };
                log::info!(
                    "Using fallback device: {}",
                    i8_array_to_string(&props.device_name)
                );
                Ok(fallback)
            }
            _ => Err(anyhow!("no suitable physical devices found")),
        }?;
        log::info!("{:?}", queue_indices);
        unsafe {
            Ok(Self {
                physical_device,
                queue_indices: queue_indices.expect("Failed to find queue indices."),
                properties: instance
                    .instance
                    .get_physical_device_properties(physical_device),
                features: instance
                    .instance
                    .get_physical_device_features(physical_device),
                memory_properties: instance
                    .instance
                    .get_physical_device_memory_properties(physical_device),
            })
        }
    }

    fn find_queue_family_indices(
        instance: &Instance,
        surface: &Surface,
        device: &vk::PhysicalDevice,
    ) -> anyhow::Result<QueueIndices> {
        let queue_props = unsafe {
            instance
                .instance
                .get_physical_device_queue_family_properties(*device)
        };
        let mut graphics_queue = None;
        let mut compute_queue = None;
        let mut transfer_queue = None;
        queue_props.iter().enumerate().for_each(|(ind, prop)| {
            if graphics_queue.is_none()
                && prop.queue_flags & vk::QueueFlags::GRAPHICS == vk::QueueFlags::GRAPHICS
            {
                if let Ok(true) =
                    surface.get_physical_device_surface_support_khr(*device, ind as u32)
                {
                    graphics_queue = Some(ind);
                }
            } else if compute_queue.is_none()
                && prop.queue_flags & vk::QueueFlags::COMPUTE == vk::QueueFlags::COMPUTE
            {
                compute_queue = Some(ind);
            } else if transfer_queue.is_none()
                && prop.queue_flags & vk::QueueFlags::TRANSFER == vk::QueueFlags::TRANSFER
            {
                transfer_queue = Some(ind);
            }
        });

        let graphics_queue_ind =
            graphics_queue.expect("Failed to find a queue index that supports graphics.") as u32;

        let compute_queue_ind = if let Some(compute_queue_ind) = compute_queue {
            Ok(compute_queue_ind as u32)
        } else if queue_props[graphics_queue_ind as usize].queue_flags & vk::QueueFlags::COMPUTE
            == vk::QueueFlags::COMPUTE
        {
            Ok(graphics_queue_ind)
        } else {
            Err(anyhow!("Failed to find a compute queue."))
        }?;

        let transfer_queue_ind = if let Some(transfer_queue_ind) = transfer_queue {
            Ok(transfer_queue_ind as u32)
        } else if queue_props[graphics_queue_ind as usize].queue_flags & vk::QueueFlags::TRANSFER
            == vk::QueueFlags::TRANSFER
        {
            Ok(graphics_queue_ind)
        } else if queue_props[compute_queue_ind as usize].queue_flags & vk::QueueFlags::TRANSFER
            == vk::QueueFlags::TRANSFER
        {
            Ok(compute_queue_ind)
        } else {
            Err(anyhow!("Failed to find a tranfer queue."))
        }?;
        Ok(QueueIndices {
            graphics: graphics_queue_ind,
            compute: compute_queue_ind,
            transfer: transfer_queue_ind,
        })
    }

    pub fn get_required_memory_index(
        &self,
        memory_requirements: vk::MemoryRequirements,
        required_memory_flags: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<u32> {
        Ok(self
            .memory_properties
            .memory_types
            .iter()
            .enumerate()
            .position(|(ind, mem_type)| {
                mem_type.property_flags.contains(required_memory_flags)
                    && (memory_requirements.memory_type_bits & (1 << ind)) != 0
            })
            .context("failed to find a suitable memory type index")? as u32)
    }
}
