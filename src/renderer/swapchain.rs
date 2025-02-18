use super::{physical_device::PhysicalDevice, surface::Surface};
use ash::{
    khr,
    vk::{self, Handle},
    Instance,
};
use std::sync::{Arc, OnceLock};

static SWAPCHAIN_DEVICE_FNS: OnceLock<khr::swapchain::Device> = OnceLock::new();
const OPTIMAL_FRAMES_IN_FLIGHT: u32 = 3;

#[derive(Clone)]
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub surface: Surface,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,

    pub num_frames_in_flight: u32,
}

// TODO: OldSwapchain
impl Swapchain {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &ash::Device,
        surface: Surface,
    ) -> anyhow::Result<Arc<Self>> {
        SWAPCHAIN_DEVICE_FNS.get_or_init(|| khr::swapchain::Device::new(instance, device));

        let swapchain = Arc::new(Self {
            swapchain: vk::SwapchainKHR::default(),
            surface,
            images: vec![],
            image_views: vec![],
            extent: vk::Extent2D::default(),
            num_frames_in_flight: OPTIMAL_FRAMES_IN_FLIGHT,
        });
        let swapchain = swapchain.recreate(physical_device, device)?;
        Ok(swapchain)
    }

    pub fn recreate(
        self: &Arc<Self>,
        physical_device: &PhysicalDevice,
        device: &ash::Device,
    ) -> anyhow::Result<Arc<Self>> {
        if !self.swapchain.is_null() {
            unsafe {
                for ind in 0..self.images.len() {
                    device.destroy_image_view(self.image_views[ind], None);
                }
                Self::get_device_fns().destroy_swapchain(self.swapchain, None);
            }
        }

        let surface_capabilities = self
            .surface
            .get_physical_device_surface_capabilities_khr(physical_device.physical_device)?;
        let extent = surface_capabilities.current_extent;

        println!("{:?}", extent);
        let (swapchain, images, image_views, num_frames_in_flight) =
            Self::create_swapchain_resources(device, physical_device, &self.surface, extent)?;

        Ok(Arc::new(Self {
            swapchain,
            images,
            image_views,
            extent,
            num_frames_in_flight,
            surface: self.surface,
        }))
    }

    pub fn acquire_next_image(
        &self,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> anyhow::Result<(u32, bool)> {
        unsafe {
            let (image_index, is_suboptimal) = Self::get_device_fns().acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore,
                fence,
            )?;

            Ok((image_index, is_suboptimal))
        }
    }

    pub fn queue_present(
        &self,
        queue: vk::Queue,
        present_info: &vk::PresentInfoKHR,
    ) -> anyhow::Result<bool> {
        unsafe { Ok(Self::get_device_fns().queue_present(queue, present_info)?) }
    }

    pub fn image(&self, image_index: usize) -> vk::Image {
        self.images[image_index]
    }

    pub fn image_view(&self, image_index: usize) -> vk::ImageView {
        self.image_views[image_index]
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            for ind in 0..self.images.len() {
                device.destroy_image_view(self.image_views[ind], None);
            }
            Self::get_device_fns().destroy_swapchain(self.swapchain, None);
            self.surface.destroy();
        }
    }

    fn create_swapchain_resources(
        device: &ash::Device,
        physical_device: &PhysicalDevice,
        surface: &Surface,
        extent: vk::Extent2D,
    ) -> anyhow::Result<(vk::SwapchainKHR, Vec<vk::Image>, Vec<vk::ImageView>, u32)> {
        unsafe {
            let surface_capabilities = physical_device.surface_capabilities;
            let image_count = u32::max(
                surface_capabilities.min_image_count,
                u32::min(
                    surface_capabilities.max_image_count,
                    OPTIMAL_FRAMES_IN_FLIGHT,
                ),
            );
            log::info!("Using {} frames in flight", image_count);

            let create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.surface)
                .min_image_count(image_count)
                .image_format(surface.format.format)
                .image_color_space(surface.format.color_space)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .queue_family_indices(&[0])
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .image_extent(extent);
            let swapchain = Self::get_device_fns().create_swapchain(&create_info, None)?;

            let images = Self::get_device_fns().get_swapchain_images(swapchain)?;
            let image_views = images
                .iter()
                .filter_map(|image| {
                    Self::create_image_view(device, *image, surface.format.format).ok()
                })
                .collect::<Vec<_>>();

            assert!(images.len() == image_views.len());
            Ok((swapchain, images, image_views, image_count))
        }
    }

    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> anyhow::Result<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(
                vk::ComponentMapping::default()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY),
            )
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        Ok(unsafe { device.create_image_view(&create_info, None)? })
    }

    fn get_device_fns() -> &'static khr::swapchain::Device {
        SWAPCHAIN_DEVICE_FNS
            .get()
            .expect("Swapchain device fns should be initialized by now.")
    }
}
