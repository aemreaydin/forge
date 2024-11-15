use anyhow::{anyhow, Context};
use ash::{ext, khr, mvk, vk, Device, Entry, Instance};
use forge::{buffer::Buffer, surface::Surface};
use glfw::{Action, Context as GlfwContext, Key};
use std::path::Path;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
use forge::{buffer::Vertex, swapchain::Swapchain};
const VALIDATION_LAYER: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

#[cfg(any(target_os = "macos", target_os = "ios"))]
const REQUIRED_INSTANCE_EXTENSIONS: &[*const i8] = &[
    vk::EXT_METAL_SURFACE_NAME.as_ptr(),
    khr::portability_enumeration::NAME.as_ptr(),
    khr::get_physical_device_properties2::NAME.as_ptr(),
    mvk::macos_surface::NAME.as_ptr(),
];
#[cfg(target_os = "linux")]
const REQUIRED_INSTANCE_EXTENSIONS: &[*const i8] = &[khr::xlib_surface::NAME.as_ptr()];

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[
    khr::portability_subset::NAME.as_ptr(),
    // ext::shader_object::NAME.as_ptr(), TODO: Macos doesn't support this just yet
]; // TODO: Make this a setting in the application

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
        required_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        log::info!("Adding {:?}", vk::EXT_DEBUG_UTILS_NAME);
    }
    required_extensions.extend_from_slice(REQUIRED_INSTANCE_EXTENSIONS);
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
    let mut required_extensions = vec![
        vk::KHR_SWAPCHAIN_NAME.as_ptr(),
        vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr(),
    ];
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

fn create_physical_device(
    instance: &Instance,
    surface: &Surface,
) -> anyhow::Result<vk::PhysicalDevice> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    let mut selected_device: Option<vk::PhysicalDevice> = None;
    let mut fallback_device: Option<vk::PhysicalDevice> = None;
    for device in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(*device) };
        let surface_support = surface.get_physical_device_surface_support_khr(*device, 0)?; // TODO: Queue family index is hardcoded
                                                                                            // TODO: These are not being used right now
        let _features = unsafe { instance.get_physical_device_features(*device) };
        let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

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
            let props = unsafe { instance.get_physical_device_properties(device) };
            log::info!(
                "Using discrete gpu: {}",
                i8_array_to_string(&props.device_name)
            );
            Ok(device)
        }
        (_, Some(fallback)) => {
            let props = unsafe { instance.get_physical_device_properties(fallback) };
            log::info!(
                "Using fallback device: {}",
                i8_array_to_string(&props.device_name)
            );
            Ok(fallback)
        }
        _ => Err(anyhow!("no suitable physical devices found")),
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
    surface: &Surface,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let surface_formats = surface.get_physical_device_surface_formats_khr(physical_device)?;
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

fn create_fence(device: &Device) -> anyhow::Result<vk::Fence> {
    let create_info = vk::FenceCreateInfo::default();
    Ok(unsafe { device.create_fence(&create_info, None)? })
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

    Ok(unsafe { device.create_render_pass(&create_info, None)? })
}

fn load_shader<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<u32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytemuck::try_cast_slice::<u8, u32>(&bytes)?.to_vec())
}

fn create_shader_module<P: AsRef<Path>>(
    device: &Device,
    path: P,
) -> anyhow::Result<vk::ShaderModule> {
    let shader_code = load_shader(path)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    Ok(unsafe { device.create_shader_module(&create_info, None)? })
}

fn create_graphics_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
) -> anyhow::Result<vk::Pipeline> {
    let name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
    let stages = &[
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(name),
    ];

    let vertex_state = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let tesselation_state = vk::PipelineTessellationStateCreateInfo::default();
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

    let color_attachment_states = &[vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )];
    let color_blend_state =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(color_attachment_states);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .line_width(1.0)
        .front_face(vk::FrontFace::CLOCKWISE);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let dyn_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dyn_states);
    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(stages)
        .vertex_input_state(&vertex_state)
        .input_assembly_state(&input_assembly_state)
        .tessellation_state(&tesselation_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .rasterization_state(&rasterization_state)
        .viewport_state(&viewport_state)
        .dynamic_state(&dynamic_state)
        .render_pass(render_pass)
        .layout(pipeline_layout)
        .subpass(0);

    // pub layout: PipelineLayout,
    // pub render_pass: RenderPass,
    // pub subpass: u32,

    // TODO: Add pipelinecache
    unsafe {
        let pipeline_res =
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None);

        let pipeline = match pipeline_res {
            Ok(pipelines) => pipelines
                .first()
                .cloned()
                .context("failed to get a graphics pipeline"),
            Err((_, vk_result)) => Err(anyhow!(
                "failed to create pipeline with error {}",
                vk_result
            )),
        }?;

        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
        Ok(pipeline)
    }
}

fn create_pipeline_layout(device: &Device) -> anyhow::Result<vk::PipelineLayout> {
    let bindings = &[vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];
    let set_create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
    let set_layout = unsafe { device.create_descriptor_set_layout(&set_create_info, None)? };
    let set_layouts = &[set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None)? };

    unsafe { device.destroy_descriptor_set_layout(set_layout, None) };

    Ok(pipeline_layout)
}
// TODO: Macos doesn't support this but keep it for the future
// fn create_shader_object<P: AsRef<Path>>(
//     shader_object_device_fns: &ext::shader_object::Device,
//     shader_path: P,
//     shader_stage_flags: vk::ShaderStageFlags,
// ) -> anyhow::Result<vk::ShaderEXT> {
//     let vert_code = load_shader(shader_path)?;
//     let name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
//     let vert_create_info = vk::ShaderCreateInfoEXT::default()
//         .stage(shader_stage_flags)
//         .code_type(vk::ShaderCodeTypeEXT::BINARY)
//         .code(&vert_code)
//         .name(name);
//     // pub next_stage: ShaderStageFlags,
//     let shader_res = unsafe { shader_object_device_fns.create_shaders(&[vert_create_info], None) };
//     shader_res
//         .map(|shaders| {
//             shaders
//                 .first()
//                 .cloned()
//                 .context("vkCreateShadersEXT failed to return a shader")
//         })
//         .map_err(|(_, result)| {
//             anyhow!(
//                 "shader creation in {:?} failed with {}",
//                 shader_stage_flags,
//                 result
//             )
//         })?
// }
//

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_target(false)
        .format_indent(None)
        .format_timestamp_nanos()
        .init();
    let mut glfw = glfw::init_no_callbacks()?;
    glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
    glfw.window_hint(glfw::WindowHint::ScaleToMonitor(false));
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
    let debug_instance_fns = create_debug_utils(&entry, &instance, debug_info)?;

    let surface = Surface::new(&entry, &instance, &window)?;
    let physical_device = create_physical_device(&instance, &surface)?;
    let device = create_device(&instance, &physical_device)?;
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let push_desc_loader = khr::push_descriptor::Device::new(&instance, &device);

    let format = get_suitable_format(physical_device, &surface)?;
    let render_pass = create_render_pass(&device, format.format)?;

    let surface_capabilities =
        surface.get_physical_device_surface_capabilities_khr(physical_device)?;
    let mut extent = surface_capabilities.current_extent;

    let mut swapchain = Swapchain::new(
        &instance,
        &device,
        surface.surface,
        render_pass,
        format,
        extent,
    )?;

    let queue = unsafe { device.get_device_queue(0, 0) };

    let pipeline_layout = create_pipeline_layout(&device)?;

    let vert_module = create_shader_module(&device, "shaders/triangle.vert.spv")?;
    let frag_module = create_shader_module(&device, "shaders/triangle.frag.spv")?;
    let graphics_pipeline = create_graphics_pipeline(
        &device,
        render_pass,
        pipeline_layout,
        vert_module,
        frag_module,
    )?;

    let command_pool = create_command_pool(&device)?;
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool);
    let command_buffers =
        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? };

    let mut acquire_semaphore = create_semaphore(&device)?;
    let mut present_semaphore = create_semaphore(&device)?;
    let fence = create_fence(&device)?;

    let triangle = Vec::from_iter([
        Vertex {
            position: [0.0, 0.5, 0.0],
            normal: [1.0, 0.0, 0.0],
            ..Default::default()
        },
        Vertex {
            position: [0.5, -0.5, 0.0],
            normal: [0.0, 1.0, 0.0],
            ..Default::default()
        },
        Vertex {
            position: [-0.5, -0.5, 0.0],
            normal: [0.0, 0.0, 1.0],
            ..Default::default()
        },
    ]);
    let indices = vec![0, 1, 2];

    let vertex_buffer = Buffer::from_data(
        &device,
        memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        triangle,
    )?;

    let index_buffer = Buffer::from_data(
        &device,
        memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::INDEX_BUFFER,
        indices,
    )?;

    while !window.should_close() {
        let surface_capabilities =
            surface.get_physical_device_surface_capabilities_khr(physical_device)?;

        if surface_capabilities.current_extent.width != extent.width
            || surface_capabilities.current_extent.height != extent.height
        {
            unsafe {
                device.device_wait_idle()?;
            }
            extent = surface_capabilities.current_extent;
            swapchain = swapchain.recreate(surface.surface, render_pass, format, extent)?;

            unsafe {
                device.destroy_semaphore(acquire_semaphore, None);
                device.destroy_semaphore(present_semaphore, None);

                acquire_semaphore = create_semaphore(&device)?;
                present_semaphore = create_semaphore(&device)?;
            }
        }

        swapchain.acquire_next_image(acquire_semaphore, vk::Fence::null())?;

        unsafe { device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())? }

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info)?;

            // TODO: Command buffer
            // let to_transfer_barrier = vk::ImageMemoryBarrier::default()
            //     .src_access_mask(vk::AccessFlags::empty())
            //     .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            //     .old_layout(vk::ImageLayout::UNDEFINED)
            //     .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            //     .image(images[image_index as usize])
            //     .subresource_range(vk::ImageSubresourceRange {
            //         aspect_mask: vk::ImageAspectFlags::COLOR,
            //         base_mip_level: 0,
            //         level_count: 1,
            //         base_array_layer: 0,
            //         layer_count: 1,
            // })
            //     .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            //     .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            // device.cmd_pipeline_barrier(
            //     command_buffer,
            //     vk::PipelineStageFlags::TOP_OF_PIPE,
            //     vk::PipelineStageFlags::TRANSFER,
            //     vk::DependencyFlags::empty(),
            //     &[],
            //     &[],
            //     &[to_transfer_barrier],
            // );

            let clear_values = &[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.2, 0.8, 0.9, 1.0],
                },
            }];
            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .clear_values(clear_values)
                .framebuffer(swapchain.framebuffer())
                .render_area(vk::Rect2D {
                    extent: swapchain.extent,
                    offset: vk::Offset2D { x: 0, y: 0 },
                });

            let viewports = &[vk::Viewport::default()
                .x(0.0)
                .y(swapchain.extent.height as f32)
                .width(swapchain.extent.width as f32)
                .height(-(swapchain.extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0)];
            device.cmd_set_viewport(command_buffer, 0, viewports);
            let scissors = &[vk::Rect2D {
                extent: swapchain.extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            }];
            device.cmd_set_scissor(command_buffer, 0, scissors);
            // render_pass: RenderPass::default(),
            // framebuffer: Framebuffer::default(),
            // render_area: Rect2D::default(),
            // clear_value_count: u32::default(),
            // p_clear_values: ::core::ptr::null(),
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let buffer_info = &[vk::DescriptorBufferInfo::default()
                .buffer(vertex_buffer.buffer)
                .offset(0)
                .range(vertex_buffer.data.len() as u64)];
            let descriptor_writes = &[vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .buffer_info(buffer_info)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)];

            push_desc_loader.cmd_push_descriptor_set(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                descriptor_writes,
            );

            device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_draw_indexed(command_buffer, index_buffer.data.len() as u32, 1, 0, 0, 0);

            // let subresource_range = vk::ImageSubresourceRange {
            //     aspect_mask: vk::ImageAspectFlags::COLOR,
            //     base_mip_level: 0,
            //     level_count: 1,
            //     base_array_layer: 0,
            //     layer_count: 1,
            // };
            // let to_present_barrier = vk::ImageMemoryBarrier::default()
            //     .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            //     .dst_access_mask(vk::AccessFlags::empty())
            //     .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            //     .new_layout(vk::ImageLayout::PRESENT_SRC_KHR) // For presenting
            //     .image(images[image_index as usize])
            //     .subresource_range(subresource_range)
            //     .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            //     .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

            // device.cmd_pipeline_barrier(
            //     command_buffer,
            //     vk::PipelineStageFlags::TRANSFER,
            //     vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            //     vk::DependencyFlags::empty(),
            //     &[],
            //     &[],
            //     &[to_present_barrier],
            // );

            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;

            let cmds = &[command_buffer];
            let waits = &[acquire_semaphore];
            let presents = &[present_semaphore];
            let images = &[swapchain.image_index()];
            let stage_flags = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(cmds)
                .wait_semaphores(waits)
                .signal_semaphores(presents)
                .wait_dst_stage_mask(stage_flags);
            device.queue_submit(queue, &[submit_info], fence)?;

            let swapchains = &[swapchain.swapchain()];
            let present_info = vk::PresentInfoKHR::default()
                .image_indices(images)
                .swapchains(swapchains)
                .wait_semaphores(presents);

            swapchain.queue_present(queue, &present_info)?;

            let fences = &[fence];
            device.wait_for_fences(fences, true, u64::MAX)?;
            device.reset_fences(fences)?;
        };

        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }
    }

    unsafe {
        device.device_wait_idle()?;

        device.free_memory(vertex_buffer.memory, None);
        device.destroy_buffer(vertex_buffer.buffer, None);

        device.free_memory(index_buffer.memory, None);
        device.destroy_buffer(index_buffer.buffer, None);

        device.destroy_fence(fence, None);
        device.destroy_semaphore(acquire_semaphore, None);
        device.destroy_semaphore(present_semaphore, None);

        device.free_command_buffers(command_pool, &command_buffers);
        device.destroy_command_pool(command_pool, None);

        device.destroy_pipeline(graphics_pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        swapchain.destroy();
        device.destroy_render_pass(render_pass, None);

        device.destroy_device(None);

        if let Some((debug_fns, debug_callback)) = debug_instance_fns {
            debug_fns.destroy_debug_utils_messenger(debug_callback, None);
        }
        surface.destroy();
        instance.destroy_instance(None);
    }
    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        glfw::WindowEvent::Key(Key::Space, _, Action::Press, _) => {}
        _ => {}
    }
}
