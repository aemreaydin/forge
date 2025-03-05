use anyhow::Context;
use ash::{
    khr,
    vk::{self, ClearValue},
};
use forge::{
    camera::LookAtCamera,
    load_image,
    renderer::{
        buffer::Vertex, image::Image, instance::Instance, shader_object::ShaderObject,
        vulkan_context::VulkanContext,
    },
    scene::{
        mesh::Mesh,
        model::{Model, Transform},
        texture::Texture,
    },
    ui::imgui_renderer::ImguiVulkanRenderer,
};
use nalgebra_glm::{Vec2, Vec3, Vec4};
use sdl3::{
    event::{Event, WindowEvent},
    keyboard::Keycode,
};
use std::sync::Arc;
use tobj::{LoadOptions, Model as TObjModel};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

struct SyncHandles {
    acquire: vk::Semaphore,
    present: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncHandles {
    pub fn new(device: &ash::Device, is_fence_signaled: bool) -> anyhow::Result<Self> {
        let acquire = forge::create_semaphore(device)?;
        let present = forge::create_semaphore(device)?;
        let fence = forge::create_fence(
            device,
            if is_fence_signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            },
        )?;
        Ok(Self {
            acquire,
            present,
            fence,
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.acquire, None);
            device.destroy_semaphore(self.present, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

pub fn load_model(model: &TObjModel) -> (Vec<Vertex>, Vec<u32>) {
    let mesh = &model.mesh;

    let mut vertices = Vec::new();
    let indices = mesh.indices.clone();

    for vtx in 0..mesh.positions.len() / 3 {
        vertices.push(Vertex {
            position: Vec4::new(
                mesh.positions[3 * vtx],
                mesh.positions[3 * vtx + 1],
                mesh.positions[3 * vtx + 2],
                1.0,
            ),
            // TODO: Normals are 0 right now
            normal: Vec4::zeros(),
            tex_coords: Vec2::new(mesh.texcoords[2 * vtx], 1.0 - mesh.texcoords[2 * vtx + 1]),
            ..Default::default()
        });
    }
    (vertices, indices)
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
        .build()?;
    video_subsystem.text_input().start(&window);

    let entry = unsafe { ash::Entry::load()? };
    let instance = Instance::new(&entry, VALIDATION_ENABLED)?;
    let surface = window.vulkan_create_surface(instance.instance.handle())?;
    let mut vulkan_context = VulkanContext::new(entry, instance, surface)?;

    let mut imgui_renderer = ImguiVulkanRenderer::new(&vulkan_context)?;

    let push_desc_loader =
        khr::push_descriptor::Device::new(vulkan_context.instance(), vulkan_context.device());
    let render_pass = forge::create_render_pass(
        vulkan_context.device(),
        vulkan_context.surface_format().format,
        vk::AttachmentLoadOp::CLEAR,
        vk::ImageLayout::UNDEFINED,
        true,
    )?;

    let push_constant_ranges = &[vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<nalgebra_glm::Mat4>() as u32)];
    let bindings = &[
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let descriptor_set_layout = forge::create_descriptor_set_layout(
        vulkan_context.device(),
        bindings,
        vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
    )?;
    let pipeline_layout = forge::create_pipeline_layout(
        vulkan_context.device(),
        push_constant_ranges,
        &[descriptor_set_layout],
    )?;
    let descriptor_pool = forge::create_descriptor_pool(
        vulkan_context.device(),
        &[vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 100,
        }],
    )?;

    let command_pool = forge::create_command_pool(
        vulkan_context.device(),
        vulkan_context.physical_device.queue_indices.graphics,
        vk::CommandPoolCreateFlags::TRANSIENT,
    )?;
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool);
    let command_buffers = unsafe {
        vulkan_context
            .device()
            .allocate_command_buffers(&command_buffer_allocate_info)?
    };

    let (vert_shader, frag_shader) = {
        log::info!("Using Shader Object");
        let push_constant_ranges = &[vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<nalgebra_glm::Mat4>() as u32)];
        let vert_shader = ShaderObject::new(
            vulkan_context.instance(),
            vulkan_context.device(),
            c"main",
            vk::ShaderStageFlags::VERTEX,
            &forge::load_shader::<_, u8>("shaders/triangle.vert.spv")?,
            &[descriptor_set_layout],
            push_constant_ranges,
        )?;
        let frag_shader = ShaderObject::new(
            vulkan_context.instance(),
            vulkan_context.device(),
            c"main",
            vk::ShaderStageFlags::FRAGMENT,
            &forge::load_shader::<_, u8>("shaders/triangle.frag.spv")?,
            &[descriptor_set_layout],
            push_constant_ranges,
        )?;

        (vert_shader, frag_shader)
    };

    let mut depth_image =
        create_depth_resources(&vulkan_context, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let mut framebuffers = vulkan_context
        .swapchain()
        .image_views()
        .iter()
        .filter_map(|image_view| {
            let image_views = &[*image_view, depth_image.image_view];
            forge::create_framebuffer(
                vulkan_context.device(),
                render_pass,
                image_views,
                vulkan_context.swapchain_extent().width,
                vulkan_context.swapchain_extent().height,
            )
            .ok()
        })
        .collect::<Vec<_>>();
    assert!(framebuffers.len() == vulkan_context.swapchain().image_views().len());

    let mut sync_handles = SyncHandles::new(vulkan_context.device(), false)?;

    let (models, _materials) = tobj::load_obj(
        mesh_path,
        &LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;

    let (vertices, indices) = load_model(&models.first().context("Failed to load model.")?.clone());
    let mesh = Mesh::new(&vulkan_context, vertices, indices)?;
    let mut cube_model = Model::new(
        "Spot".to_string(),
        vec![Arc::new(mesh)],
        Some(Transform {
            scale: Vec3::new(1.0, 1.0, 1.0),
            ..Default::default()
        }),
    );
    let cube_image = load_image("meshes/spot_texture.png")?;
    let cube_texture = Texture::from_2d_data(&vulkan_context, cube_image)?;
    let cube_sampler = forge::create_sampler(vulkan_context.device())?;

    let mut camera = LookAtCamera::new(
        Vec3::new(0.0, 0.0, -2.0),
        cube_model.transform.position,
        45.0,
        vulkan_context.swapchain_extent().width as f32,
        vulkan_context.swapchain_extent().height as f32,
        0.1,
        100.0,
    );

    let mut last_tick = std::time::Instant::now();

    let mut resized = false;
    let mut event_pump = sdl_context.event_pump()?;
    'main_loop: loop {
        let now = std::time::Instant::now();
        let delta_time = now.duration_since(last_tick);
        last_tick = now;

        for event in event_pump.poll_iter() {
            imgui_renderer.process_event(&event);
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'main_loop,
                Event::Window {
                    win_event: WindowEvent::Resized(..),
                    ..
                } => {
                    resized = true;
                }
                Event::MouseMotion {
                    mousestate,
                    xrel,
                    yrel,
                    ..
                } => {
                    if mousestate.middle() {
                        camera.orbit(xrel, yrel, delta_time.as_secs_f32());
                    }
                }
                Event::MouseWheel { y, .. } => {
                    camera.zoom(y, delta_time.as_secs_f32());
                }
                _ => {}
            }
        }

        //check_resize(
        //    &vulkan_context,
        //    format,
        //    render_pass,
        //    &mut extent,
        //    &mut swapchain,
        //    &mut acquire_semaphore,
        //    &mut present_semaphore,
        //)?;
        if resized {
            log::info!("Resizing");
            vulkan_context.resized()?;
            unsafe {
                framebuffers.iter().for_each(|framebuffer| {
                    vulkan_context
                        .device()
                        .destroy_framebuffer(*framebuffer, None);
                });
                depth_image.destroy(vulkan_context.device());
            }

            depth_image =
                create_depth_resources(&vulkan_context, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            framebuffers = vulkan_context
                .swapchain()
                .image_views()
                .iter()
                .filter_map(|image_view| {
                    let image_views = &[*image_view, depth_image.image_view];
                    log::info!(
                        "Creating framebuffers with {}-{}",
                        vulkan_context.swapchain_extent().width,
                        vulkan_context.swapchain_extent().height
                    );
                    forge::create_framebuffer(
                        vulkan_context.device(),
                        render_pass,
                        image_views,
                        vulkan_context.swapchain_extent().width,
                        vulkan_context.swapchain_extent().height,
                    )
                    .ok()
                })
                .collect::<Vec<_>>();
            sync_handles = recreate_sync_handles(vulkan_context.device(), sync_handles)?;
            imgui_renderer.resized(vulkan_context.swapchain())?;
            resized = false;
            continue;
        }

        let (image_index, _is_suboptimal) = vulkan_context
            .swapchain()
            .acquire_next_image(sync_handles.acquire, vk::Fence::null())?;
        let image_index = image_index as usize;

        unsafe {
            vulkan_context
                .device()
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?
        }

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            vulkan_context
                .device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            // TODO: Change barrier functionality when using a graphics pipeline
            let to_transfer_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(vulkan_context.swapchain().image(image_index))
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
            vulkan_context.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_barrier],
            );

            let clear_values = [
                ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.2, 0.8, 0.9, 1.0],
                    },
                },
                ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let viewports = &[vk::Viewport::default()
                .x(0.0)
                .y(vulkan_context.swapchain_extent().height as f32)
                .width(vulkan_context.swapchain_extent().width as f32)
                .height(-(vulkan_context.swapchain_extent().height as f32))
                .min_depth(0.0)
                .max_depth(1.0)];
            vulkan_context
                .device()
                .cmd_set_viewport(command_buffer, 0, viewports);
            let scissors = &[vk::Rect2D {
                extent: vulkan_context.swapchain_extent(),
                offset: vk::Offset2D { x: 0, y: 0 },
            }];
            vulkan_context
                .device()
                .cmd_set_scissor(command_buffer, 0, scissors);

            let model =
                nalgebra_glm::scale(&nalgebra_glm::Mat4::identity(), &cube_model.transform.scale);
            let model = nalgebra_glm::rotate_x(&model, cube_model.transform.rotation.x);
            let model = nalgebra_glm::rotate_y(&model, cube_model.transform.rotation.y);
            let model = nalgebra_glm::rotate_z(&model, cube_model.transform.rotation.z);
            let model = nalgebra_glm::translate(&model, &cube_model.transform.position);
            let mvp = camera.view_projection() * model;

            let color_attachments = &[vk::RenderingAttachmentInfo::default()
                .image_view(vulkan_context.swapchain().image_view(image_index))
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_values[0])];
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_image.image_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_values[1]);
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vulkan_context.swapchain_extent(),
                })
                .layer_count(1)
                .color_attachments(color_attachments)
                .depth_attachment(&depth_attachment);

            vulkan_context
                .device()
                .cmd_begin_rendering(command_buffer, &rendering_info);
            vert_shader.bind_shader(command_buffer, &[vk::ShaderStageFlags::VERTEX]);
            frag_shader.bind_shader(command_buffer, &[vk::ShaderStageFlags::FRAGMENT]);
            ShaderObject::set_vertex_input(command_buffer, &[], &[]);
            ShaderObject::set_dynamic_state(
                vulkan_context.device(),
                command_buffer,
                viewports,
                scissors,
            );

            if cube_model.visible {
                cube_model.meshes.iter().for_each(|mesh| {
                    let buffer_info = &[vk::DescriptorBufferInfo::default()
                        .buffer(mesh.vertex_buffer.buffer)
                        .offset(0)
                        .range(mesh.vertex_buffer.size)];
                    let image_info = &[vk::DescriptorImageInfo::default()
                        .image_view(cube_texture.image.image_view)
                        .sampler(cube_sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                    let descriptor_writes = &[
                        vk::WriteDescriptorSet::default()
                            .descriptor_count(1)
                            .dst_binding(0)
                            .buffer_info(buffer_info)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER),
                        vk::WriteDescriptorSet::default()
                            .descriptor_count(1)
                            .dst_binding(1)
                            .image_info(image_info)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                    ];
                    push_desc_loader.cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        descriptor_writes,
                    );
                    vulkan_context.device().cmd_bind_index_buffer(
                        command_buffer,
                        mesh.index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    vulkan_context.device().cmd_push_constants(
                        command_buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, // same as above
                        0,
                        std::slice::from_raw_parts(
                            mvp.as_ptr() as *const u8,
                            std::mem::size_of::<nalgebra_glm::Mat4>(),
                        ),
                    );
                    vulkan_context.device().cmd_draw_indexed(
                        command_buffer,
                        mesh.indices.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );
                });
            }

            vulkan_context.device().cmd_end_rendering(command_buffer);
            vulkan_context.device().end_command_buffer(command_buffer)?;

            imgui_renderer.start_frame(
                vulkan_context.swapchain_extent(),
                &window,
                &event_pump,
                image_index,
            )?;
            imgui_renderer.draw(|ui| {
                if let Some(wnd) = ui
                    .window("Forge")
                    .size([300.0, 400.0], imgui::Condition::FirstUseEver)
                    .begin()
                {
                    ui.slider("Camera Zoom Speed", 1.0, 10.0, &mut camera.zoom_speed);
                    ui.slider("Camera Orbit Speed", 1.0, 60.0, &mut camera.orbit_speed);
                    if ui.button("Reset Camera Settings") {
                        camera.reset_settings();
                    }

                    ui.checkbox(&cube_model.name, &mut cube_model.visible);

                    wnd.end();
                }
            })?;
            imgui_renderer.end_frame(vulkan_context.swapchain().image(image_index))?;

            let cmds = &[command_buffer, imgui_renderer.current_command_buffer];
            let waits = &[sync_handles.acquire];
            let presents = &[sync_handles.present];
            let image_indices = &[image_index as u32];
            let stage_flags = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(cmds)
                .wait_semaphores(waits)
                .signal_semaphores(presents)
                .wait_dst_stage_mask(stage_flags);
            vulkan_context.device().queue_submit(
                vulkan_context.device.graphics_queue,
                &[submit_info],
                sync_handles.fence,
            )?;

            let swapchains = &[vulkan_context.swapchain().swapchain];
            let present_info = vk::PresentInfoKHR::default()
                .image_indices(image_indices)
                .swapchains(swapchains)
                .wait_semaphores(presents);

            vulkan_context
                .swapchain()
                .queue_present(vulkan_context.device.graphics_queue, &present_info)?;

            let fences = &[sync_handles.fence];
            vulkan_context
                .device()
                .wait_for_fences(fences, true, u64::MAX)?;
            vulkan_context.device().reset_fences(fences)?;
        };
    }

    unsafe {
        vulkan_context.device().device_wait_idle()?;
        imgui_renderer.destroy();

        depth_image.destroy(vulkan_context.device());
        sync_handles.destroy(vulkan_context.device());

        cube_texture.destroy(&vulkan_context);
        vulkan_context.device().destroy_sampler(cube_sampler, None);

        framebuffers.iter().for_each(|framebuffer| {
            vulkan_context
                .device()
                .destroy_framebuffer(*framebuffer, None);
        });

        cube_model.meshes.iter().for_each(|mesh| {
            vulkan_context
                .device()
                .free_memory(mesh.vertex_buffer.memory, None);
            vulkan_context
                .device()
                .destroy_buffer(mesh.vertex_buffer.buffer, None);

            vulkan_context
                .device()
                .free_memory(mesh.index_buffer.memory, None);
            vulkan_context
                .device()
                .destroy_buffer(mesh.index_buffer.buffer, None);
        });
        vulkan_context
            .device()
            .free_command_buffers(command_pool, &command_buffers);
        vulkan_context
            .device()
            .destroy_command_pool(command_pool, None);

        vert_shader.destroy();
        frag_shader.destroy();

        vulkan_context
            .device()
            .destroy_descriptor_pool(descriptor_pool, None);
        vulkan_context
            .device()
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        vulkan_context
            .device()
            .destroy_pipeline_layout(pipeline_layout, None);

        vulkan_context
            .device()
            .destroy_render_pass(render_pass, None);
    }
    Ok(())
}
fn create_depth_resources(
    vulkan_context: &VulkanContext,
    required_memory_flags: vk::MemoryPropertyFlags,
) -> anyhow::Result<Image> {
    let vk::Extent2D { width, height } = vulkan_context.swapchain_extent();
    let image_create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::D32_SFLOAT)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = Image::new(
        &vulkan_context.physical_device,
        &vulkan_context.device.device,
        required_memory_flags,
        image_create_info,
        vk::ImageViewType::TYPE_2D,
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .layer_count(1)
            .level_count(1),
    )?;

    Ok(image)
}

fn recreate_sync_handles(
    device: &ash::Device,
    sync_handles: SyncHandles,
) -> anyhow::Result<SyncHandles> {
    sync_handles.destroy(device);
    SyncHandles::new(device, false)
}

// TODO: Check this code, doesn't look good
//fn check_resize(
//    vulkan_context: &VulkanContext,
//    format: vk::SurfaceFormatKHR,
//    render_pass: vk::RenderPass,
//    extent: &mut vk::Extent2D,
//    mut swapchain: Swapchain,
//    acquire_semaphore: &mut vk::Semaphore,
//    present_semaphore: &mut vk::Semaphore,
//) -> anyhow::Result<()> {
//let surface_capabilities = vulkan_context
//    .surface
//    .get_physical_device_surface_capabilities_khr(vulkan_context.physical_device())?;
//if surface_capabilities.current_extent.width != extent.width
//    || surface_capabilities.current_extent.height != extent.height
//{
//    unsafe {
//        vulkan_context.device().device_wait_idle()?;
//    }
//    *extent = surface_capabilities.current_extent;
//    swapchain = swapchain.new(vulkan_context.device(), vulkan_context.physical_device())?;
//
//    unsafe {
//        vulkan_context
//            .device()
//            .destroy_semaphore(*acquire_semaphore, None);
//        vulkan_context
//            .device()
//            .destroy_semaphore(*present_semaphore, None);
//
//        *acquire_semaphore = create_semaphore(vulkan_context.device())?;
//        *present_semaphore = create_semaphore(vulkan_context.device())?;
//    }
//};
//    Ok(())
//}
