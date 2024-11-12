use ash::{khr, vk, Device, Instance};

struct SwapchainResources {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}
pub struct Swapchain {
    handles: SwapchainResources,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,

    current_image_index: u32,

    device: Device,
    loader: khr::swapchain::Device,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
    ) -> anyhow::Result<Self> {
        let loader = khr::swapchain::Device::new(instance, device);
        let handles =
            Self::create_swapchain(device, &loader, surface, render_pass, format, extent)?;
        Ok(Self {
            handles,

            format,
            extent,

            current_image_index: 0,

            device: device.clone(),
            loader,
        })
    }

    pub fn recreate(
        self,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
    ) -> anyhow::Result<Self> {
        self.destroy();

        let handles = Self::create_swapchain(
            &self.device,
            &self.loader,
            surface,
            render_pass,
            format,
            extent,
        )?;
        Ok(Self {
            handles,

            format,
            extent,

            current_image_index: 0,

            device: self.device,
            loader: self.loader,
        })
    }

    pub fn acquire_next_image(
        &mut self,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> anyhow::Result<bool> {
        unsafe {
            let (image_index, is_suboptimal) = self.loader.acquire_next_image(
                self.handles.swapchain,
                u64::MAX,
                semaphore,
                fence,
            )?;

            self.current_image_index = image_index;
            Ok(is_suboptimal)
        }
    }

    pub fn queue_present(
        &self,
        queue: vk::Queue,
        present_info: &vk::PresentInfoKHR,
    ) -> anyhow::Result<bool> {
        unsafe { Ok(self.loader.queue_present(queue, present_info)?) }
    }

    pub fn swapchain(&self) -> vk::SwapchainKHR {
        self.handles.swapchain
    }

    pub fn image_index(&self) -> u32 {
        self.current_image_index
    }

    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.handles.framebuffers[self.current_image_index as usize]
    }

    pub fn image(&self) -> vk::Image {
        self.handles.images[self.current_image_index as usize]
    }

    pub fn destroy(&self) {
        unsafe {
            for ind in 0..self.handles.images.len() {
                self.device
                    .destroy_image_view(self.handles.image_views[ind], None);
                self.device
                    .destroy_framebuffer(self.handles.framebuffers[ind], None);
            }
            self.loader.destroy_swapchain(self.handles.swapchain, None);
        }
    }

    fn create_swapchain(
        device: &Device,
        loader: &khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
    ) -> anyhow::Result<SwapchainResources> {
        unsafe {
            let create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(3)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .queue_family_indices(&[0])
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .image_extent(extent);
            let swapchain = loader.create_swapchain(&create_info, None)?;

            let images = loader.get_swapchain_images(swapchain)?;
            let mut image_views = Vec::with_capacity(images.len());
            let mut framebuffers = Vec::with_capacity(images.len());
            for (ind, image) in images.iter().enumerate() {
                image_views.push(Self::create_image_view(device, *image, format.format)?);
                framebuffers.push(Self::create_framebuffer(
                    device,
                    render_pass,
                    image_views[ind],
                    extent.width,
                    extent.height,
                )?);
            }
            Ok(SwapchainResources {
                swapchain,
                images,
                image_views,
                framebuffers,
            })
        }
    }

    fn create_image_view(
        device: &Device,
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

    fn create_framebuffer(
        device: &Device,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        width: u32,
        height: u32,
    ) -> anyhow::Result<vk::Framebuffer> {
        let attachments = &[image_view];
        let create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .width(width)
            .height(height)
            .layers(1)
            .attachments(attachments);
        Ok(unsafe { device.create_framebuffer(&create_info, None)? })
    }
}
