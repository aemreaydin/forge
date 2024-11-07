use anyhow::{anyhow, Context};
use ash::{
    ext, khr, mvk,
    vk::{self, EXT_DEBUG_UTILS_NAME, KHR_SWAPCHAIN_NAME},
    Device, Entry, Instance,
};
use glfw::{Action, Context as GlfwContext, Key};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };
#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_INSTANCE_EXTENSIONS: &[*const i8] = &[
    vk::EXT_METAL_SURFACE_NAME.as_ptr(),
    khr::portability_enumeration::NAME.as_ptr(),
    khr::get_physical_device_properties2::NAME.as_ptr(),
    mvk::macos_surface::NAME.as_ptr(),
];
#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[khr::portability_subset::NAME.as_ptr()];

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_name = if callback_data.p_message_id_name.is_null() {
        std::borrow::Cow::from("")
    } else {
        std::ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let formatted_message = format!(
        "\n[{:?}] [{}] ({}): {}\n",
        message_type, message_id_name, callback_data.message_id_number, message
    );

    match message_severity {
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) => {
            log::error!("{}", formatted_message);
        }
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) => {
            log::warn!("{}", formatted_message);
        }
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) => {
            log::info!("{}", formatted_message);
        }
        s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE) => {
            log::debug!("{}", formatted_message);
        }
        _ => log::trace!("{}", formatted_message),
    }

    vk::FALSE
}

fn i8_array_to_string(slice: &[i8]) -> String {
    String::from_utf8_lossy(
        &slice
            .iter()
            .take_while(|&&c| c != 0) // Stop at null terminator
            .map(|&c| c as u8)
            .collect::<Vec<u8>>(),
    )
    .to_string()
}

fn create_instance(
    entry: &Entry,
    debug_info: &mut vk::DebugUtilsMessengerCreateInfoEXT,
) -> anyhow::Result<Instance> {
    let app_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"forge\0") };
    let layers = get_required_layers(entry)?;
    let extensions = get_required_instance_extensions(entry)?;
    let version = unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);

    let appinfo = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(version);

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&appinfo)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(create_flags);
    if VALIDATION_ENABLED {
        create_info = create_info.push_next(debug_info);
    }

    Ok(unsafe { entry.create_instance(&create_info, None)? })
}

fn create_debug_utils(
    entry: &Entry,
    instance: &Instance,
    debug_info: vk::DebugUtilsMessengerCreateInfoEXT,
) -> anyhow::Result<Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>> {
    if !VALIDATION_ENABLED {
        return Ok(None);
    }

    let debug_utils_loader = ext::debug_utils::Instance::new(entry, instance);
    let debug_call_back =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None)? };
    Ok(Some((debug_utils_loader, debug_call_back)))
}

fn get_required_layers(entry: &Entry) -> anyhow::Result<Vec<*const i8>> {
    let layer_names = unsafe {
        entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|layer| std::ffi::CStr::from_ptr(layer.layer_name.as_ptr()))
            .collect::<Vec<_>>()
    };
    let layers = {
        if VALIDATION_ENABLED && layer_names.contains(&VALIDATION_LAYER) {
            [VALIDATION_LAYER].iter().map(|l| l.as_ptr()).collect()
        } else {
            vec![]
        }
    };
    Ok(layers)
}

fn get_required_instance_extensions(entry: &Entry) -> anyhow::Result<Vec<*const i8>> {
    let mut required_extensions = vec![khr::surface::NAME.as_ptr()];
    if VALIDATION_ENABLED {
        required_extensions.push(EXT_DEBUG_UTILS_NAME.as_ptr());
        log::info!("Adding {:?}", EXT_DEBUG_UTILS_NAME);
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    required_extensions.extend_from_slice(MACOS_REQUIRED_INSTANCE_EXTENSIONS);
    let system_extension_names = unsafe {
        entry
            .enumerate_instance_extension_properties(None)?
            .iter()
            .map(|extension| std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()))
            .collect::<Vec<_>>()
    };
    for required_extension in &required_extensions {
        let extension_name = unsafe { std::ffi::CStr::from_ptr(*required_extension) };
        if !system_extension_names.contains(&extension_name) {
            return Err(anyhow!(
                "extension {} not supported by the system",
                extension_name.to_string_lossy()
            ));
        }
    }
    Ok(required_extensions)
}

fn get_required_device_extensions(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> anyhow::Result<Vec<*const i8>> {
    let mut required_extensions = vec![KHR_SWAPCHAIN_NAME.as_ptr()];
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    required_extensions.extend_from_slice(MACOS_REQUIRED_DEVICE_EXTENSIONS);

    let device_extension_names = unsafe {
        instance
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

fn create_surface(
    entry: &Entry,
    instance: &Instance,
    handle: &(impl HasDisplayHandle + HasWindowHandle),
) -> anyhow::Result<(khr::surface::Instance, vk::SurfaceKHR)> {
    let raw_display_handle = handle
        .display_handle()
        .expect("failed to get raw display handle")
        .as_raw();
    let raw_window_handle = handle
        .window_handle()
        .expect("failed to get raw window handle")
        .as_raw();
    unsafe {
        Ok((
            khr::surface::Instance::new(entry, instance),
            ash_window::create_surface(
                entry,
                instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?,
        ))
    }
}

fn create_physical_device(
    instance: &Instance,
    surface_instance_fns: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<vk::PhysicalDevice> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    let mut selected_device: Option<vk::PhysicalDevice> = None;
    for device in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(*device) };
        let surface_support = unsafe {
            surface_instance_fns.get_physical_device_surface_support(*device, 0, surface)?
        }; // TODO: Queue family index is hardcoded
           // TODO: These are not being used right now
        let _features = unsafe { instance.get_physical_device_features(*device) };
        let _is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

        match surface_support {
            true => {
                if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    log::info!("Using {}", i8_array_to_string(&props.device_name));
                    return Ok(*device);
                } else {
                    selected_device = Some(*device)
                }
            }
            false => {}
        }
    }

    match selected_device {
        Some(device) => {
            let props = unsafe { instance.get_physical_device_properties(device) };
            log::info!("Using {}", i8_array_to_string(&props.device_name));
            Ok(device)
        }
        None => Err(anyhow!("no suitable physical devices found")),
    }
}

fn create_device(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> anyhow::Result<Device> {
    let extensions = get_required_device_extensions(instance, physical_device)?;
    let queue_create_infos = [vk::DeviceQueueCreateInfo {
        queue_count: 1,
        queue_family_index: 0,
        ..Default::default()
    }
    .queue_priorities(&[1.0])];
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extensions);
    Ok(unsafe { instance.create_device(*physical_device, &create_info, None)? })
}

fn get_suitable_format(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_instance_fns: &khr::surface::Instance,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let surface_formats = unsafe {
        surface_instance_fns.get_physical_device_surface_formats(physical_device, surface)
    }?;
    surface_formats
        .into_iter()
        .find(|format| {
            // TODO: Not the best way to do it
            !(format.color_space != vk::ColorSpaceKHR::SRGB_NONLINEAR
                || format.format != vk::Format::R8G8B8A8_SRGB
                    && format.format != vk::Format::B8G8R8A8_SRGB)
        })
        .context("failed to find a suitable surface format")
}

fn create_swapchain(
    instance: &Instance,
    device: &Device,
    surface: vk::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    width: u32,
    height: u32,
) -> anyhow::Result<(khr::swapchain::Device, vk::SwapchainKHR)> {
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(3)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .queue_family_indices(&[0])
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .image_extent(vk::Extent2D { width, height });

    let swapchain_device_fns = khr::swapchain::Device::new(instance, device);
    let swapchain = unsafe { swapchain_device_fns.create_swapchain(&create_info, None)? };

    Ok((swapchain_device_fns, swapchain))
}

fn create_command_pool(device: &Device) -> anyhow::Result<vk::CommandPool> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(0)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT); // TODO: Queue family index hard coded

    Ok(unsafe { device.create_command_pool(&create_info, None)? })
}

fn create_semaphore(device: &Device) -> anyhow::Result<vk::Semaphore> {
    let create_info = vk::SemaphoreCreateInfo::default();
    Ok(unsafe { device.create_semaphore(&create_info, None)? })
}

fn create_render_pass(device: &Device, format: vk::Format) -> anyhow::Result<vk::RenderPass> {
    let color_desc = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let descs = &[color_desc];

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let refs = &[color_ref];

    let color_subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(refs);
    let subpasses = &[color_subpass];

    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(descs)
        .subpasses(subpasses);

    Ok(unsafe { device.create_render_pass(&create_info, None) }?)
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_target(false)
        .format_indent(None)
        .format_timestamp_nanos()
        .init();
    let mut glfw = glfw::init_no_callbacks()?;
    let (mut window, events) = glfw
        .create_window(1920, 1080, "forge", glfw::WindowMode::Windowed)
        .context("failed to create a glfw window")?;
    window.set_key_polling(true);
    window.make_current();

    let entry = unsafe { Entry::load()? };

    let mut debug_info = if VALIDATION_ENABLED {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));
        debug_info
    } else {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
    };

    let instance = create_instance(&entry, &mut debug_info)?;
    let debug_handles = create_debug_utils(&entry, &instance, debug_info)?;

    let (surface_instance_fns, surface) = create_surface(&entry, &instance, &window)?;
    let physical_device = create_physical_device(&instance, &surface_instance_fns, surface)?;
    let format = get_suitable_format(physical_device, surface, &surface_instance_fns)?;

    let device = create_device(&instance, &physical_device)?;
    let queue = unsafe { device.get_device_queue(0, 0) };

    let (swapchain_device_fns, swapchain) = create_swapchain(
        &instance,
        &device,
        surface,
        format,
        window.get_size().0 as u32,
        window.get_size().1 as u32,
    )?;
    let images = unsafe { swapchain_device_fns.get_swapchain_images(swapchain)? };

    let render_pass = create_render_pass(&device, format.format)?;

    let command_pool = create_command_pool(&device)?;
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool);
    let command_buffers =
        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? };

    let acquire_semaphore = create_semaphore(&device)?;
    let present_semaphore = create_semaphore(&device)?;

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        let (image_index, is_suboptimal) = unsafe {
            swapchain_device_fns.acquire_next_image(
                swapchain,
                u64::MAX,
                acquire_semaphore,
                vk::Fence::null(),
            )?
        };

        unsafe { device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())? }

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info)?; // TODO: Command buffer
            let to_transfer_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(images[image_index as usize])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_barrier],
            );

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            device.cmd_clear_color_image(
                command_buffer,
                images[image_index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [1.0, 1.0, 0.0, 1.0],
                },
                &[subresource_range],
            );
            let to_present_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR) // For presenting
                .image(images[image_index as usize])
                .subresource_range(subresource_range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present_barrier],
            );

            device.end_command_buffer(command_buffer)?;

            let cmds = &[command_buffer];
            let waits = &[acquire_semaphore];
            let presents = &[present_semaphore];
            let images = &[image_index];
            let stage_flags = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(cmds)
                .wait_semaphores(waits)
                .signal_semaphores(presents)
                .wait_dst_stage_mask(stage_flags);
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;

            let swapchains = &[swapchain];
            let present_info = vk::PresentInfoKHR::default()
                .image_indices(images)
                .swapchains(swapchains)
                .wait_semaphores(presents);
            swapchain_device_fns.queue_present(queue, &present_info)?;

            device.device_wait_idle()?;
        };
    }

    unsafe {
        device.device_wait_idle()?;

        device.destroy_semaphore(acquire_semaphore, None);
        device.destroy_semaphore(present_semaphore, None);

        device.free_command_buffers(command_pool, &command_buffers);
        device.destroy_command_pool(command_pool, None);

        device.destroy_render_pass(render_pass, None);
        swapchain_device_fns.destroy_swapchain(swapchain, None);
        surface_instance_fns.destroy_surface(surface, None);
        device.destroy_device(None);

        if let Some((debug_fns, debug_callback)) = debug_handles {
            debug_fns.destroy_debug_utils_messenger(debug_callback, None);
        }
        instance.destroy_instance(None);
    }
    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        _ => {}
    }
}
