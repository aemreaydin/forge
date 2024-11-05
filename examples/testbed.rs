use anyhow::Context;
use ash::{ext::metal_objects, khr, mvk, prelude::VkResult, vk, Entry, Instance};
use glfw::{Action, Context as GlfwContext, Key};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

fn create_instance(entry: &Entry) -> VkResult<Instance> {
    let app_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"forge\0") };
    let layers = unsafe {
        let layer_names = [std::ffi::CStr::from_bytes_with_nul_unchecked(
            b"VK_LAYER_KHRONOS_validation\0",
        )];
        layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>()
    };

    let mut extensions = vec![];
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extensions.push(vk::EXT_METAL_SURFACE_NAME.as_ptr());
        extensions.push(khr::surface::NAME.as_ptr());
        extensions.push(khr::portability_enumeration::NAME.as_ptr());
        extensions.push(khr::get_physical_device_properties2::NAME.as_ptr());
        extensions.push(mvk::macos_surface::NAME.as_ptr());
    }

    let appinfo = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&appinfo)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(create_flags);

    unsafe { entry.create_instance(&create_info, None) }
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

fn create_surface(
    entry: &Entry,
    instance: &Instance,
    handle: &(impl HasDisplayHandle + HasWindowHandle),
) -> anyhow::Result<vk::SurfaceKHR> {
    let raw_display_handle = handle
        .display_handle()
        .expect("failed to get raw display handle")
        .as_raw();
    let raw_window_handle = handle
        .window_handle()
        .expect("failed to get raw window handle")
        .as_raw();
    unsafe {
        Ok(ash_window::create_surface(
            entry,
            instance,
            raw_display_handle,
            raw_window_handle,
            None,
        )?)
    }
}

fn create_physical_device(
    entry: &Entry,
    instance: &Instance,
) -> anyhow::Result<vk::PhysicalDevice> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    let surface_loader = khr::surface::Instance::new(entry, instance);
    for device in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(*device) };
        let features = unsafe { instance.get_physical_device_features(*device) };
        let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            log::info!("Using {}", i8_array_to_string(&props.device_name));
            return Ok(*device);
        }
    }

    let first_device = physical_devices
        .first()
        .cloned()
        .context("no suitable physical device found")?;
    let props = unsafe { instance.get_physical_device_properties(first_device) };
    log::info!("Using {}", i8_array_to_string(&props.device_name));
    Ok(first_device)
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
    let instance = create_instance(&entry)?;
    let physical_device = create_physical_device(&entry, &instance);
    let surface = create_surface(&entry, &instance, &window)?;

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }
    }
    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        _ => {}
    }
}
