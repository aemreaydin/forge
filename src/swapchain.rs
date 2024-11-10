use ash::{khr, vk, Device, Instance};

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,

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
        let (swapchain, images, image_views, framebuffers) =
            Self::create_swapchain(device, &loader, surface, render_pass, format, extent)?;
        Ok(Self {
            swapchain,
            images,
            image_views,
            framebuffers,

            format,
            extent,

            current_image_index: 0,

            device: device.clone(),
            loader,
        })
    }

    pub fn recreate_swapchain(
        self,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
    ) -> anyhow::Result<Self> {
        self.destroy_swapchain();

        let (swapchain, images, image_views, framebuffers) = Self::create_swapchain(
            &self.device,
            &self.loader,
            surface,
            render_pass,
            format,
            extent,
        )?;
        Ok(Self {
            swapchain,
            images,
            image_views,
            framebuffers,

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
            let (image_index, is_suboptimal) =
                self.loader
                    .acquire_next_image(self.swapchain, u64::MAX, semaphore, fence)?;

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

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn image_index(&self) -> u32 {
        self.current_image_index
    }

    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffers[self.current_image_index as usize]
    }

    pub fn image(&self) -> vk::Image {
        self.images[self.current_image_index as usize]
    }

    pub fn destroy_swapchain(&self) {
        unsafe {
            for ind in 0..self.images.len() {
                self.device.destroy_image_view(self.image_views[ind], None);
                self.device
                    .destroy_framebuffer(self.framebuffers[ind], None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }

    fn create_swapchain(
        device: &Device,
        loader: &khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        format: vk::SurfaceFormatKHR,
        extent: vk::Extent2D,
    ) -> anyhow::Result<(
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        Vec<vk::Framebuffer>,
    )> {
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
            Ok((swapchain, images, image_views, framebuffers))
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
