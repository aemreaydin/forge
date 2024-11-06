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
        "[{:?}] [{}] ({}): {}\n",
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

fn create_swapchain(
    instance: &Instance,
    device: &Device,
    surface: vk::SurfaceKHR,
    surface_formats: &[vk::SurfaceFormatKHR],
    width: u32,
    height: u32,
) -> anyhow::Result<(khr::swapchain::Device, vk::SwapchainKHR)> {
    let surface_format = surface_formats
        .iter()
        .find(|format| {
            // TODO: Not the best way to do it
            !(format.color_space != vk::ColorSpaceKHR::SRGB_NONLINEAR
                || format.format != vk::Format::R8G8B8A8_SRGB
                    && format.format != vk::Format::B8G8R8A8_SRGB)
        })
        .context("failed to find a suitable surface format")?;
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(3)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .queue_family_indices(&[0])
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .image_extent(vk::Extent2D { width, height });

    let swapchain_device_fns = khr::swapchain::Device::new(instance, device);
    let swapchain = unsafe { swapchain_device_fns.create_swapchain(&create_info, None)? };

    Ok((swapchain_device_fns, swapchain))
}

fn create_semaphore(device: &Device) -> anyhow::Result<vk::Semaphore> {
    let create_info = vk::SemaphoreCreateInfo::default();
    Ok(unsafe { device.create_semaphore(&create_info, None)? })
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

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
    let surface_formats = unsafe {
        surface_instance_fns.get_physical_device_surface_formats(physical_device, surface)?
    };

    let device = create_device(&instance, &physical_device)?;
    let (swapchain_device_fns, swapchain) = create_swapchain(
        &instance,
        &device,
        surface,
        &surface_formats,
        window.get_size().0 as u32,
        window.get_size().1 as u32,
    )?;

    let waitSemaphore = create_semaphore(&device)?;
    let presentSemaphore = create_semaphore(&device)?;

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }
    }

    unsafe {
        device.device_wait_idle()?;

        if let Some((debug_fns, debug_callback)) = debug_handles {
            debug_fns.destroy_debug_utils_messenger(debug_callback, None);
        }
        device.destroy_semaphore(waitSemaphore, None);
        device.destroy_semaphore(presentSemaphore, None);
        swapchain_device_fns.destroy_swapchain(swapchain, None);
        surface_instance_fns.destroy_surface(surface, None);
        device.destroy_device(None);
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
