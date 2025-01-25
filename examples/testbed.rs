use crate::model::load_model;
use anyhow::{anyhow, Context};
use ash::{khr, vk};
use either::Either;
use forge::{
    buffer::{Buffer, Vertex},
    device::Device,
    physical_device::PhysicalDevice,
    shader_object::Shader,
    surface::Surface,
    swapchain::Swapchain,
};
use nalgebra_glm::Vec3;
use sdl3::{event::Event, keyboard::Keycode};
use std::path::Path;
use tobj::LoadOptions;

mod model;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

fn get_suitable_format(
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> anyhow::Result<vk::SurfaceFormatKHR> {
    let surface_formats = surface.get_physical_device_surface_formats_khr(physical_device)?;
    surface_formats
        .into_iter()
        .find(|format| {
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

fn load_shader<P: AsRef<Path>, T: bytemuck::Pod>(path: P) -> anyhow::Result<Vec<T>> {
    let bytes = std::fs::read(path)?;
    Ok(bytemuck::try_cast_slice::<u8, T>(&bytes)
        .expect("Failed to cast shader to u8.")
        .to_vec())
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
    let name = c"main";
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace"))
        .format_target(false)
        .format_indent(None)
        .format_timestamp_nanos()
        .init();

    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem
        .window("forge", 1920, 1080)
        .position_centered()
        .vulkan()
        .resizable()
        .build()
        .unwrap();
    let mut event_pump = sdl_context.event_pump()?;

    let instance = forge::instance::Instance::new(VALIDATION_ENABLED)?;
    let surface = Surface::new(instance.clone(), &window)?;
    let physical_device = PhysicalDevice::new(&instance, &surface)?;
    let device = Device::new(&instance, &physical_device.handle)?;

    let push_desc_loader = khr::push_descriptor::Device::new(&instance.instance, &device.handle);
    let format = get_suitable_format(physical_device.handle, &surface)?;
    let render_pass = create_render_pass(&device.handle, format.format)?;

    let surface_capabilities =
        surface.get_physical_device_surface_capabilities_khr(physical_device.handle)?;
    let mut extent = surface_capabilities.current_extent;
    let mut swapchain = Swapchain::new(
        &instance.instance,
        &device.handle,
        surface.surface,
        render_pass,
        physical_device.memory_properties,
        format,
        extent,
    )?;

    let queue = unsafe { device.handle.get_device_queue(0, 0) };

    let (pipeline_layout, descriptor_set_layout) = create_pipeline_layout(&device.handle)?;

    let either_pipeline_or_objects = match device.device_support.shader_ext {
        true => {
            log::info!("Using Shader Object");
            let push_constant_ranges = &[vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(size_of::<nalgebra_glm::Mat4>() as u32)];
            let vert_shader = forge::shader_object::Shader::new(
                &instance.instance,
                &device.handle,
                c"main",
                vk::ShaderStageFlags::VERTEX,
                &load_shader::<_, u8>("shaders/triangle.vert.spv")?,
                &[descriptor_set_layout],
                push_constant_ranges,
            )?;
            let frag_shader = forge::shader_object::Shader::new(
                &instance.instance,
                &device.handle,
                c"main",
                vk::ShaderStageFlags::FRAGMENT,
                &load_shader::<_, u8>("shaders/triangle.frag.spv")?,
                &[descriptor_set_layout],
                push_constant_ranges,
            )?;

            Either::Left((vert_shader, frag_shader))
        }
        false => {
            log::info!("Using Graphics Pipeline");
            let vert_module = create_shader_module(&device.handle, "shaders/triangle.vert.spv")?;
            let frag_module = create_shader_module(&device.handle, "shaders/triangle.frag.spv")?;
            Either::Right(create_graphics_pipeline(
                &device.handle,
                render_pass,
                pipeline_layout,
                vert_module,
                frag_module,
            )?)
        }
    };

    let command_pool = create_command_pool(&device.handle)?;
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool);
    let command_buffers = unsafe {
        device
            .handle
            .allocate_command_buffers(&command_buffer_allocate_info)?
    };

    let mut acquire_semaphore = create_semaphore(&device.handle)?;
    let mut present_semaphore = create_semaphore(&device.handle)?;
    let fence = create_fence(&device.handle)?;

    let (models, _materials) = tobj::load_obj(
        mesh_path,
        &LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;

    let (vertices, indices) = load_model(&models.first().context("Failed to load model.")?.clone());

    let vertex_buffer = Buffer::from_data(
        &device.handle,
        physical_device.memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vertices,
    )?;

    let index_buffer = Buffer::from_data(
        &device.handle,
        physical_device.memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::INDEX_BUFFER,
        indices,
    )?;

    let start_time = std::time::Instant::now();
    while !handle_window_event(&mut event_pump) {
        let surface_capabilities =
            surface.get_physical_device_surface_capabilities_khr(physical_device.handle)?;

        if surface_capabilities.current_extent.width != extent.width
            || surface_capabilities.current_extent.height != extent.height
        {
            unsafe {
                device.handle.device_wait_idle()?;
            }
            extent = surface_capabilities.current_extent;
            swapchain = swapchain.recreate(
                surface.surface,
                render_pass,
                physical_device.memory_properties,
                format,
                extent,
            )?;

            unsafe {
                device.handle.destroy_semaphore(acquire_semaphore, None);
                device.handle.destroy_semaphore(present_semaphore, None);

                acquire_semaphore = create_semaphore(&device.handle)?;
                present_semaphore = create_semaphore(&device.handle)?;
            }
        }

        swapchain.acquire_next_image(acquire_semaphore, vk::Fence::null())?;

        unsafe {
            device
                .handle
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?
        }

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .handle
                .begin_command_buffer(command_buffer, &begin_info)?;

            let to_transfer_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(swapchain.image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            device.handle.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_barrier],
            );

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

            let viewports = &[vk::Viewport::default()
                .x(0.0)
                .y(swapchain.extent.height as f32)
                .width(swapchain.extent.width as f32)
                .height(-(swapchain.extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0)];
            device.handle.cmd_set_viewport(command_buffer, 0, viewports);
            let scissors = &[vk::Rect2D {
                extent: swapchain.extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            }];
            device.handle.cmd_set_scissor(command_buffer, 0, scissors);

            let buffer_info = &[vk::DescriptorBufferInfo::default()
                .buffer(vertex_buffer.buffer)
                .offset(0)
                .range(vertex_buffer.size)];
            let descriptor_writes = &[vk::WriteDescriptorSet::default()
                .descriptor_count(1)
                .dst_binding(0)
                .buffer_info(buffer_info)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)];

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
            match &either_pipeline_or_objects {
                Either::Left((vert_shader, frag_shader)) => {
                    let color_attachments = &[vk::RenderingAttachmentInfo::default()
                        .image_view(swapchain.image_view())
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(clear_values[0])];
                    let depth_attachment = vk::RenderingAttachmentInfo::default()
                        .image_view(swapchain.depth_image_view())
                        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(clear_values[1]);
                    let rendering_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent,
                        })
                        .layer_count(1)
                        .color_attachments(color_attachments)
                        .depth_attachment(&depth_attachment);
                    device
                        .handle
                        .cmd_begin_rendering(command_buffer, &rendering_info);
                    vert_shader.bind_shader(command_buffer, &[vk::ShaderStageFlags::VERTEX]);
                    frag_shader.bind_shader(command_buffer, &[vk::ShaderStageFlags::FRAGMENT]);
                    Shader::set_vertex_input(command_buffer, &[], &[]);
                    Shader::set_dynamic_state(&device.handle, command_buffer, viewports, scissors);
                    push_desc_loader.cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        descriptor_writes,
                    );
                    device.handle.cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.handle.cmd_push_constants(
                        command_buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, // same as above
                        0,
                        std::slice::from_raw_parts(
                            mvp.as_ptr() as *const u8,
                            std::mem::size_of::<nalgebra_glm::Mat4>(),
                        ),
                    );
                    device.handle.cmd_draw_indexed(
                        command_buffer,
                        index_buffer.data.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );
                    device.handle.cmd_end_rendering(command_buffer);
                }
                Either::Right(graphics_pipeline) => {
                    let render_pass_begin = vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass)
                        .clear_values(clear_values)
                        .framebuffer(swapchain.framebuffer())
                        .render_area(vk::Rect2D {
                            extent: swapchain.extent,
                            offset: vk::Offset2D { x: 0, y: 0 },
                        });
                    device.handle.cmd_begin_render_pass(
                        command_buffer,
                        &render_pass_begin,
                        vk::SubpassContents::INLINE,
                    );
                    push_desc_loader.cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        descriptor_writes,
                    );

                    device.handle.cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.handle.cmd_push_constants(
                        command_buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, // same as above
                        0,
                        std::slice::from_raw_parts(
                            mvp.as_ptr() as *const u8,
                            std::mem::size_of::<nalgebra_glm::Mat4>(),
                        ),
                    );
                    device.handle.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *graphics_pipeline,
                    );
                    device.handle.cmd_draw_indexed(
                        command_buffer,
                        index_buffer.data.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );
                    device.handle.cmd_end_render_pass(command_buffer);
                }
            }

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_present_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR) // For presenting
                .image(swapchain.image())
                .subresource_range(subresource_range)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

            device.handle.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present_barrier],
            );

            device.handle.end_command_buffer(command_buffer)?;

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
            device.handle.queue_submit(queue, &[submit_info], fence)?;

            let swapchains = &[swapchain.swapchain()];
            let present_info = vk::PresentInfoKHR::default()
                .image_indices(images)
                .swapchains(swapchains)
                .wait_semaphores(presents);

            swapchain.queue_present(queue, &present_info)?;

            let fences = &[fence];
            device.handle.wait_for_fences(fences, true, u64::MAX)?;
            device.handle.reset_fences(fences)?;
        };
    }

    unsafe {
        device.handle.device_wait_idle()?;

        device.handle.free_memory(vertex_buffer.memory, None);
        device.handle.destroy_buffer(vertex_buffer.buffer, None);

        device.handle.free_memory(index_buffer.memory, None);
        device.handle.destroy_buffer(index_buffer.buffer, None);

        device.handle.destroy_fence(fence, None);
        device.handle.destroy_semaphore(acquire_semaphore, None);
        device.handle.destroy_semaphore(present_semaphore, None);

        device
            .handle
            .free_command_buffers(command_pool, &command_buffers);
        device.handle.destroy_command_pool(command_pool, None);

        match either_pipeline_or_objects {
            Either::Left((vert_shader, frag_shader)) => {
                vert_shader.destroy();
                frag_shader.destroy();
            }
            Either::Right(graphics_pipeline) => {
                device.handle.destroy_pipeline(graphics_pipeline, None);
            }
        }
        device
            .handle
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.handle.destroy_pipeline_layout(pipeline_layout, None);

        swapchain.destroy();
        device.handle.destroy_render_pass(render_pass, None);

        device.handle.destroy_device(None);
    }
    surface.destroy();
    instance.destroy();
    Ok(())
}

fn handle_window_event(event_pump: &mut sdl3::EventPump) -> bool {
    for event in event_pump.poll_iter() {
        match event {
            Event::Quit { .. }
            | Event::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            } => return true,
            _ => {}
        }
    }
    false
}
