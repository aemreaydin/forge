use anyhow::{anyhow, Context};
use ash::{khr, mvk, vk};
use core::f32;
use forge::{
    buffer::{Buffer, Vertex},
    instance::Instance,
    surface::Surface,
    swapchain::Swapchain,
};
use glfw::{Action, Context as GlfwContext, Key};
use nalgebra_glm::{Vec2, Vec3, Vec4};
use std::path::Path;
use tobj::LoadOptions;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

#[cfg(any(target_os = "macos", target_os = "ios"))]
const REQUIRED_INSTANCE_EXTENSIONS: &[&std::ffi::CStr] = &[
    vk::EXT_METAL_SURFACE_NAME,
    khr::portability_enumeration::NAME,
    khr::get_physical_device_properties2::NAME,
    mvk::macos_surface::NAME,
];
#[cfg(target_os = "linux")]
const REQUIRED_INSTANCE_EXTENSIONS: &[&std::ffi::CStr] = &[khr::xlib_surface::NAME];
#[cfg(target_os = "windows")]
const REQUIRED_INSTANCE_EXTENSIONS: &[&std::ffi::CStr] = &[khr::win32_surface::NAME];

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MACOS_REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[
    khr::portability_subset::NAME.as_ptr(),
    // ext::shader_object::NAME.as_ptr(), TODO: Macos doesn't support this just yet
]; // TODO: Make this a setting in the application

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

fn get_required_device_extensions(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> anyhow::Result<Vec<*const i8>> {
    #[allow(unused_mut)]
    let mut required_extensions = vec![
        vk::KHR_SWAPCHAIN_NAME.as_ptr(),
        vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr(),
    ];
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    required_extensions.extend_from_slice(MACOS_REQUIRED_DEVICE_EXTENSIONS);

    let device_extension_names = unsafe {
        instance
            .instance
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
    let physical_devices = unsafe { instance.instance.enumerate_physical_devices()? };

    let mut selected_device: Option<vk::PhysicalDevice> = None;
    let mut fallback_device: Option<vk::PhysicalDevice> = None;
    for device in &physical_devices {
        let props = unsafe { instance.instance.get_physical_device_properties(*device) };
        let surface_support = surface.get_physical_device_surface_support_khr(*device, 0)?; // TODO: Queue family index is hardcoded
                                                                                            // TODO: These are not being used right now
        let _features = unsafe { instance.instance.get_physical_device_features(*device) };
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
    }
}

fn create_device(
    instance: &Instance,
    physical_device: &vk::PhysicalDevice,
) -> anyhow::Result<ash::Device> {
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
    Ok(unsafe {
        instance
            .instance
            .create_device(*physical_device, &create_info, None)?
    })
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

fn create_command_pool(device: &ash::Device) -> anyhow::Result<vk::CommandPool> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(0)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT); // TODO: Queue family index hard coded

    Ok(unsafe { device.create_command_pool(&create_info, None)? })
}

fn create_semaphore(device: &ash::Device) -> anyhow::Result<vk::Semaphore> {
    let create_info = vk::SemaphoreCreateInfo::default();
    Ok(unsafe { device.create_semaphore(&create_info, None)? })
}

fn create_fence(device: &ash::Device) -> anyhow::Result<vk::Fence> {
    let create_info = vk::FenceCreateInfo::default();
    Ok(unsafe { device.create_fence(&create_info, None)? })
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> anyhow::Result<vk::RenderPass> {
    let color_desc = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let depth_desc = vk::AttachmentDescription::default()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let descs = &[color_desc, depth_desc];

    let color_refs = &[vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let depth_refs = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_refs)
        .depth_stencil_attachment(&depth_refs);
    let subpasses = &[color_subpass];
    let dependencies = &[vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(descs)
        .subpasses(subpasses)
        .dependencies(dependencies);

    Ok(unsafe { device.create_render_pass(&create_info, None)? })
}

fn load_shader<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<u32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytemuck::try_cast_slice::<u8, u32>(&bytes)?.to_vec())
}

fn create_shader_module<P: AsRef<Path>>(
    device: &ash::Device,
    path: P,
) -> anyhow::Result<vk::ShaderModule> {
    let shader_code = load_shader(path)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    Ok(unsafe { device.create_shader_module(&create_info, None)? })
}

fn create_graphics_pipeline(
    device: &ash::Device,
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
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_write_enable(true)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0);

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
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .cull_mode(vk::CullModeFlags::BACK);

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

fn create_pipeline_layout(
    device: &ash::Device,
) -> anyhow::Result<(vk::PipelineLayout, vk::DescriptorSetLayout)> {
    let push_constant_ranges = &[vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<nalgebra_glm::Mat4>() as u32)];
    let bindings = &[vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];
    let set_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
    let set_layout = unsafe { device.create_descriptor_set_layout(&set_create_info, None)? };
    let set_layouts = &[set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None)? };

    Ok((pipeline_layout, set_layout))
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
pub fn normalize_mesh(vertices: &mut [Vertex]) {
    let mut positions = vertices.iter_mut().map(|v| v.position).collect::<Vec<_>>();
    if positions.is_empty() {
        return;
    }

    // Find bounding box
    let mut min = positions[0];
    let mut max = positions[0];

    for vertex in positions.iter() {
        min = nalgebra_glm::min2(&min, vertex);
        max = nalgebra_glm::max2(&max, vertex);
    }

    // Calculate center and scale
    let center = (max + min) * 0.5;
    let scale = (max - min).max() * 0.5;

    // Normalize vertices
    for vertex in positions.iter_mut() {
        *vertex = (*vertex - center) / scale;
    }
    vertices
        .iter_mut()
        .enumerate()
        .for_each(|(ind, v)| v.position = positions[ind]);
}
fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .context("Failed to get a mesh path from args")?;
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

    let required_extensions = [
        std::slice::from_ref(&khr::surface::NAME),
        REQUIRED_INSTANCE_EXTENSIONS,
    ]
    .concat();
    let instance = forge::instance::Instance::new(&required_extensions, VALIDATION_ENABLED)?;

    let surface = Surface::new(instance.clone(), &window)?;
    let physical_device = create_physical_device(&instance, &surface)?;
    let device = create_device(&instance, &physical_device)?;
    let memory_properties = unsafe {
        instance
            .instance
            .get_physical_device_memory_properties(physical_device)
    };

    let push_desc_loader = khr::push_descriptor::Device::new(&instance.instance, &device);

    let format = get_suitable_format(physical_device, &surface)?;
    let render_pass = create_render_pass(&device, format.format)?;

    let surface_capabilities =
        surface.get_physical_device_surface_capabilities_khr(physical_device)?;
    let mut extent = surface_capabilities.current_extent;

    let mut swapchain = Swapchain::new(
        &instance.instance,
        &device,
        surface.surface,
        render_pass,
        memory_properties,
        format,
        extent,
    )?;

    let queue = unsafe { device.get_device_queue(0, 0) };

    let (pipeline_layout, descriptor_set_layout) = create_pipeline_layout(&device)?;

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

    let (models, _materials) = tobj::load_obj(
        mesh_path,
        &LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;

    let (vertices, indices) = {
        let model = models.first().context("Failed to get the model")?;
        let mesh = &model.mesh;

        let mut vertices = Vec::new();
        let indices = mesh.indices.clone();

        let positions = mesh.positions.as_slice();
        let normals = mesh.normals.as_slice();
        let texcoords = mesh.texcoords.as_slice();

        let vertex_count = positions.len() / 3;

        for i in 0..vertex_count {
            let position = Vec4::new(
                positions[i * 3],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
                1.0,
            );
            let normal = if !mesh.normals.is_empty() {
                Vec4::new(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2], 1.0)
            } else {
                [0.0, 0.0, 0.0, 1.0].into()
            };
            let tex_coords = if !mesh.texcoords.is_empty() {
                Vec2::new(texcoords[i * 3], texcoords[i * 3 + 1])
            } else {
                [0.0, 0.0].into()
            };
            let vertex = Vertex {
                position,
                normal,
                tex_coords,
                ..Default::default()
            };
            vertices.push(vertex);
        }
        (vertices, indices)
    };
    // normalize_mesh(&mut vertices);

    let vertex_buffer = Buffer::from_data(
        &device,
        memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vertices,
    )?;

    let index_buffer = Buffer::from_data(
        &device,
        memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::INDEX_BUFFER,
        indices,
    )?;

    let start_time = std::time::Instant::now();
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
            swapchain = swapchain.recreate(
                surface.surface,
                render_pass,
                memory_properties,
                format,
                extent,
            )?;

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

            let clear_values = &[
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.2, 0.8, 0.9, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
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
                .range(vertex_buffer.size)];
            let descriptor_writes = &[vk::WriteDescriptorSet::default()
                .descriptor_count(1)
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

            let time = start_time.elapsed().as_secs_f32();
            let model = nalgebra_glm::rotate(
                &nalgebra_glm::Mat4::identity(),
                time,
                &Vec3::new(1.0, 1.0, 0.0),
            );
            let projection = nalgebra_glm::perspective_fov_rh_zo(
                f32::to_radians(45.0),
                extent.width as f32,
                extent.height as f32,
                0.01,
                10.0,
            );
            let view = nalgebra_glm::look_at_rh(
                &Vec3::new(0.0, 0.0, -1.0),
                &Vec3::default(),
                &Vec3::new(0.0, 1.0, 0.0),
            );
            let mvp = projection * view * model;
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, // same as above
                0,
                std::slice::from_raw_parts(
                    mvp.as_ptr() as *const u8,
                    std::mem::size_of::<nalgebra_glm::Mat4>(),
                ),
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
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_pipeline_layout(pipeline_layout, None);

        swapchain.destroy();
        device.destroy_render_pass(render_pass, None);

        device.destroy_device(None);
    }
    surface.destroy();
    instance.destroy();
    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        glfw::WindowEvent::Key(Key::Space, _, Action::Press, _) => {}
        _ => {}
    }
}
