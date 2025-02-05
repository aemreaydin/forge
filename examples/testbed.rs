use anyhow::{anyhow, Context};
use ash::{
    khr,
    vk::{self, ClearValue, Handle},
};
use either::Either;
use forge::{
    renderer::{
        buffer::{Buffer, Vertex},
        context::VulkanContext,
        image::Image,
        instance::Instance,
        shader_object::Shader,
        surface::Surface,
        swapchain::Swapchain,
    },
    ui::imgui_sdl3_binding::Sdl3Binding,
};
use imgui::sys::{ImDrawIdx, ImDrawVert};
use imgui::TextureId;
use nalgebra_glm::{Vec2, Vec3, Vec4};
use sdl3::{event::Event, keyboard::Keycode};
use std::{mem::offset_of, path::Path};
use tobj::{LoadOptions, Model};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub fn load_model(model: &Model) -> (Vec<Vertex>, Vec<u32>) {
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
}
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

fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
) -> anyhow::Result<vk::CommandPool> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);

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

fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    initial_image_layout: vk::ImageLayout,
    is_using_depth: bool,
) -> anyhow::Result<vk::RenderPass> {
    let color_desc = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(initial_image_layout)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let mut descs = vec![color_desc];

    if is_using_depth {
        descs.push(
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
    }

    let color_refs = &[vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let mut color_subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_refs);

    let depth_refs = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    if is_using_depth {
        color_subpass = color_subpass.depth_stencil_attachment(&depth_refs);
    }
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
        .attachments(&descs)
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

// TODO: Add cullmode
fn create_graphics_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    vertex_state: vk::PipelineVertexInputStateCreateInfo,
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
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let tesselation_state = vk::PipelineTessellationStateCreateInfo::default();
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_attachment_states = &[vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
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
        .cull_mode(vk::CullModeFlags::NONE);

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

fn create_imgui_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> anyhow::Result<vk::PipelineLayout> {
    let push_constant_ranges = &[vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        // Scale and Offset for 2D Imgui
        .size(2 * size_of::<nalgebra_glm::Vec2>() as u32)];
    let set_layouts = &[set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None)? };

    Ok(pipeline_layout)
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

fn create_descriptor_set_layout(
    vulkan_context: &VulkanContext,
    descriptor_type: vk::DescriptorType,
    stage_flags: vk::ShaderStageFlags,
) -> anyhow::Result<vk::DescriptorSetLayout> {
    let bindings = &[vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(descriptor_type)
        .stage_flags(stage_flags)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
    Ok(unsafe {
        vulkan_context
            .device()
            .create_descriptor_set_layout(&create_info, None)?
    })
}

fn create_sampler(vulkan_context: &VulkanContext) -> anyhow::Result<vk::Sampler> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(-1000.0)
        .max_lod(1000.0)
        .unnormalized_coordinates(false)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .max_anisotropy(1.0);

    unsafe {
        vulkan_context
            .device()
            .create_sampler(&create_info, None)
            .context("Failed to create a sampler")
    }
}

fn create_texture_descriptor_set(
    vulkan_context: &VulkanContext,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
) -> anyhow::Result<vk::DescriptorSet> {
    let set_layouts = &[set_layout];

    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(set_layouts);

    unsafe {
        let descriptor_set = vulkan_context
            .device()
            .allocate_descriptor_sets(&allocate_info)?
            .first()
            .cloned()
            .context("Failed to allocate a descriptor set for texture")?;

        let descriptor_image_info = &[vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(image_view)
            .image_layout(image_layout)];

        let write_descs = &[vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(descriptor_image_info)];
        vulkan_context
            .device()
            .update_descriptor_sets(write_descs, &[]);

        vulkan_context
            .device()
            .destroy_descriptor_set_layout(set_layout, None);
        Ok(descriptor_set)
    }
}

fn create_descriptor_pool(
    vulkan_context: &VulkanContext,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> anyhow::Result<vk::DescriptorPool> {
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(pool_sizes)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(
            pool_sizes
                .iter()
                .fold(0, |acc, pool_size| acc + pool_size.descriptor_count),
        );
    let descriptor_pool = unsafe {
        vulkan_context
            .device()
            .create_descriptor_pool(&create_info, None)?
    };
    Ok(descriptor_pool)
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

fn imgui_render_state(
    vulkan_context: &VulkanContext,
    draw_data: &imgui::DrawData,
    pipeline_layout: vk::PipelineLayout,
    cmd: vk::CommandBuffer,
    width: f32,
    height: f32,
) -> anyhow::Result<()> {
    unsafe {
        let viewports = &[vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(width)
            .height(height)
            .min_depth(0.0)
            .max_depth(1.0)];
        vulkan_context.device().cmd_set_viewport(cmd, 0, viewports);

        let scale = draw_data.display_size.map(|v| 2.0 / v);
        let pos = draw_data
            .display_pos
            .iter()
            .enumerate()
            .map(|(i, v)| -1.0 - v * scale[i])
            .collect::<Vec<f32>>();

        let scale_bytes = bytemuck::try_cast_slice::<f32, u8>(scale.as_slice())
            .expect("Failed to cast imgui scale slice");
        let translate_bytes = bytemuck::try_cast_slice::<f32, u8>(pos.as_slice())
            .expect("Failed to cast imgui translate slice");
        vulkan_context.device().cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            scale_bytes,
        );
        vulkan_context.device().cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            (pos.len() * size_of::<f32>()) as u32,
            translate_bytes,
        );
    }
    Ok(())
}

fn align_buffer_size(size: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    (size + alignment - 1) & !(alignment - 1)
}

fn render_draw_data(
    vulkan_context: &VulkanContext,
    draw_data: &imgui::DrawData,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    cmd: vk::CommandBuffer,
) -> anyhow::Result<()> {
    let width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
    let height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
    if width <= 0.0 || height <= 0.0 {
        return Ok(());
    }

    if draw_data.total_vtx_count > 0 {
        let vtx_size = align_buffer_size(
            draw_data.total_vtx_count as u64 * size_of::<ImDrawVert>() as u64,
            256,
        );
        let idx_size = align_buffer_size(
            draw_data.total_idx_count as u64 * size_of::<ImDrawIdx>() as u64,
            256,
        );

        let vtx_buffer = Buffer::new(
            vulkan_context,
            vtx_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let vtx_ptr = vtx_buffer.map_memory(vulkan_context)?;

        let idx_buffer = Buffer::new(
            vulkan_context,
            idx_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let idx_ptr = idx_buffer.map_memory(vulkan_context)?;

        let mut vtx_vec = Vec::with_capacity(draw_data.total_vtx_count as usize);
        let mut idx_vec = Vec::with_capacity(draw_data.total_idx_count as usize);
        draw_data.draw_lists().for_each(|draw_list| {
            vtx_vec.extend_from_slice(draw_list.vtx_buffer());
            idx_vec.extend_from_slice(draw_list.idx_buffer());
        });
        unsafe {
            (vtx_vec.as_ptr() as *mut std::ffi::c_void)
                .copy_to_nonoverlapping(vtx_ptr, size_of::<ImDrawVert>() * vtx_vec.len());
            (idx_vec.as_ptr() as *mut std::ffi::c_void)
                .copy_to_nonoverlapping(idx_ptr, size_of::<ImDrawIdx>() * idx_vec.len());
        }

        let ranges = &[
            vk::MappedMemoryRange::default()
                .memory(vtx_buffer.memory)
                .size(vk::WHOLE_SIZE),
            vk::MappedMemoryRange::default()
                .memory(idx_buffer.memory)
                .size(vk::WHOLE_SIZE),
        ];
        unsafe {
            vulkan_context.device().flush_mapped_memory_ranges(ranges)?;
            idx_buffer.unmap_memory(vulkan_context.device());
            vtx_buffer.unmap_memory(vulkan_context.device());
        }
        imgui_render_state(
            vulkan_context,
            draw_data,
            pipeline_layout,
            cmd,
            width,
            height,
        )?;

        if draw_data.total_vtx_count > 0 {
            unsafe {
                vulkan_context.device().cmd_bind_index_buffer(
                    cmd,
                    idx_buffer.buffer,
                    0,
                    if size_of::<ImDrawIdx>() == 2 {
                        vk::IndexType::UINT16
                    } else {
                        vk::IndexType::UINT32
                    },
                );
                vulkan_context
                    .device()
                    .cmd_bind_vertex_buffers(cmd, 0, &[vtx_buffer.buffer], &[0]);
            }
        }

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
        let mut clip_min = [0.0, 0.0];
        let mut clip_max = [0.0, 0.0];
        let mut vtx_offset = 0;
        let mut idx_offset = 0;
        draw_data.draw_lists().for_each(|draw_list| unsafe {
            draw_list.commands().for_each(|draw_cmd| {
                // TODO: UserCallback https://github.com/ocornut/imgui/blob/e4db4e423d78bce7e6c050f1f0710b3b635a9871/backends/imgui_impl_vulkan.cpp#L580
                match draw_cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        clip_min = [
                            (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1],
                        ];
                        clip_max = [
                            (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];
                        if clip_min[0] < 0.0 {
                            clip_min[0] = 0.0;
                        }
                        if clip_min[1] < 0.0 {
                            clip_min[1] = 0.0;
                        }
                        if clip_max[0] > width {
                            clip_max[0] = width;
                        }
                        if clip_max[1] > height {
                            clip_max[1] = height;
                        }
                        // TODO: Check if this ever is an issue
                        //if clip_max[0] <= clip_min[0] || clip_max[1] <= clip_max[1] {
                        //    continue;
                        //}

                        let scissors = &[vk::Rect2D {
                            extent: vk::Extent2D {
                                width: (clip_max[0] - clip_min[0]) as u32,
                                height: (clip_max[1] - clip_min[1]) as u32,
                            },
                            offset: vk::Offset2D {
                                x: clip_min[0] as i32,
                                y: clip_min[1] as i32,
                            },
                        }];
                        vulkan_context.device().cmd_set_scissor(cmd, 0, scissors);

                        vulkan_context.device().cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );
                        vulkan_context.device().cmd_draw_indexed(
                            cmd,
                            count as u32,
                            1,
                            (cmd_params.idx_offset + idx_offset) as u32,
                            (cmd_params.vtx_offset + vtx_offset) as i32,
                            0,
                        );
                    }
                    imgui::DrawCmd::ResetRenderState => todo!(),
                    imgui::DrawCmd::RawCallback { .. } => todo!(),
                }
            });
            idx_offset += draw_list.idx_buffer().len();
            vtx_offset += draw_list.vtx_buffer().len();
        });

        let scissors = &[vk::Rect2D {
            extent: vk::Extent2D {
                width: width as u32,
                height: height as u32,
            },
            offset: vk::Offset2D { x: 0, y: 0 },
        }];
        unsafe {
            vulkan_context.device().cmd_set_scissor(cmd, 0, scissors);
        }
    }

    Ok(())
}

fn create_framebuffer(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    width: u32,
    height: u32,
) -> anyhow::Result<vk::Framebuffer> {
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .width(width)
        .height(height)
        .layers(1)
        .attachments(image_views);
    Ok(unsafe { device.create_framebuffer(&create_info, None)? })
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
    let mut event_pump = sdl_context.event_pump()?;

    let entry = unsafe { ash::Entry::load()? };
    let instance = Instance::new(&entry, VALIDATION_ENABLED)?;
    let surface = window.vulkan_create_surface(instance.instance.handle())?;
    let vulkan_context = VulkanContext::new(entry, instance, surface)?;

    let mut imgui = imgui::Context::create();

    let sdl3_imgui_binding = Sdl3Binding::new(&mut imgui);

    let push_desc_loader =
        khr::push_descriptor::Device::new(vulkan_context.instance(), vulkan_context.device());
    let format = get_suitable_format(vulkan_context.physical_device(), &vulkan_context.surface)?;
    let render_pass = create_render_pass(
        vulkan_context.device(),
        format.format,
        vk::AttachmentLoadOp::CLEAR,
        vk::ImageLayout::UNDEFINED,
        true,
    )?;

    let surface_capabilities = vulkan_context
        .surface
        .get_physical_device_surface_capabilities_khr(vulkan_context.physical_device())?;
    let mut extent = surface_capabilities.current_extent;
    let mut swapchain = Swapchain::new(
        vulkan_context.instance(),
        vulkan_context.device(),
        vulkan_context.surface(),
        render_pass,
        vulkan_context.physical_device.memory_properties,
        format,
        extent,
    )?;

    let (pipeline_layout, descriptor_set_layout) = create_pipeline_layout(vulkan_context.device())?;
    let descriptor_pool = create_descriptor_pool(
        &vulkan_context,
        &[vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 100,
        }],
    )?;

    let command_pool = create_command_pool(
        vulkan_context.device(),
        vulkan_context.physical_device.queue_indices.graphics,
    )?;
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(command_pool);
    let command_buffers = unsafe {
        vulkan_context
            .device()
            .allocate_command_buffers(&command_buffer_allocate_info)?
    };

    let imgui_set_layout = create_descriptor_set_layout(
        &vulkan_context,
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
    )?;
    let imgui_pipeline_layout =
        create_imgui_pipeline_layout(vulkan_context.device(), imgui_set_layout)?;
    let imgui_render_pass = create_render_pass(
        vulkan_context.device(),
        format.format,
        vk::AttachmentLoadOp::LOAD,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        false,
    )?;
    // TODO: The image copy operation should use transfer and graphics queues.
    let imgui_command_pool = create_command_pool(
        vulkan_context.device(),
        vulkan_context.physical_device.queue_indices.graphics,
    )?;
    let imgui_command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .command_pool(imgui_command_pool);
    let font_cmds = unsafe {
        vulkan_context
            .device()
            .allocate_command_buffers(&imgui_command_buffer_allocate_info)?
    };
    let font_cmd = font_cmds
        .first()
        .cloned()
        .context("Failed to allocate a font command buffer.")?;
    unsafe {
        vulkan_context
            .device()
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        vulkan_context
            .device()
            .begin_command_buffer(font_cmd, &begin_info)?;
    }
    // imgui
    //     .fonts()
    //     .add_font(&[imgui::FontSource::DefaultFontData {
    //         config: Default::default(),
    //     }]);
    imgui.fonts().add_font(&[imgui::FontSource::TtfData {
        data: include_bytes!("../fonts/imgui_font.ttf"),
        size_pixels: 13.0,
        config: Default::default(),
    }]);
    let fonts = imgui.fonts().build_rgba32_texture();
    let fonts_size = fonts.width * fonts.height * 4 * size_of::<u8>() as u32;
    let fonts_image = Image::new(
        &vulkan_context,
        fonts_size as u64,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::ImageCreateInfo::default()
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .image_type(vk::ImageType::TYPE_2D)
            .mip_levels(1)
            .array_layers(1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: fonts.width,
                height: fonts.height,
                depth: 1,
            })
            .tiling(vk::ImageTiling::OPTIMAL),
        vk::ImageViewType::TYPE_2D,
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1),
    )?;
    fonts_image.copy_to_host(&vulkan_context, font_cmd, fonts.data)?;

    let font_sampler = create_sampler(&vulkan_context)?;

    let font_descriptor_set = create_texture_descriptor_set(
        &vulkan_context,
        imgui_set_layout,
        descriptor_pool,
        font_sampler,
        fonts_image.image_view,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )?;
    imgui.fonts().tex_id = TextureId::new(font_descriptor_set.as_raw() as usize);
    let imgui_framebuffers = swapchain
        .image_views()
        .iter()
        .filter_map(|image_view| {
            let views = &[*image_view];
            create_framebuffer(
                vulkan_context.device(),
                imgui_render_pass,
                views,
                extent.width,
                extent.height,
            )
            .ok()
        })
        .collect::<Vec<_>>();

    let either_pipeline_or_objects = match vulkan_context.device.device_support.shader_ext {
        true => {
            log::info!("Using Shader Object");
            let push_constant_ranges = &[vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(size_of::<nalgebra_glm::Mat4>() as u32)];
            let vert_shader = Shader::new(
                vulkan_context.instance(),
                vulkan_context.device(),
                c"main",
                vk::ShaderStageFlags::VERTEX,
                &load_shader::<_, u8>("shaders/triangle.vert.spv")?,
                &[descriptor_set_layout],
                push_constant_ranges,
            )?;
            let frag_shader = Shader::new(
                vulkan_context.instance(),
                vulkan_context.device(),
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
            let vert_module =
                create_shader_module(vulkan_context.device(), "shaders/triangle.vert.spv")?;
            let frag_module =
                create_shader_module(vulkan_context.device(), "shaders/triangle.frag.spv")?;

            let imgui_vert_module =
                create_shader_module(vulkan_context.device(), "shaders/imgui.vert.spv")?;
            let imgui_frag_module =
                create_shader_module(vulkan_context.device(), "shaders/imgui.frag.spv")?;

            let binding_descs = &[vk::VertexInputBindingDescription::default()
                .stride(size_of::<ImDrawVert>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];
            let attribute_descs = &[
                vk::VertexInputAttributeDescription::default()
                    .location(0)
                    .binding(binding_descs[0].binding)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(ImDrawVert, pos) as u32),
                vk::VertexInputAttributeDescription::default()
                    .location(1)
                    .binding(binding_descs[0].binding)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(ImDrawVert, uv) as u32),
                vk::VertexInputAttributeDescription::default()
                    .location(2)
                    .binding(binding_descs[0].binding)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .offset(offset_of!(ImDrawVert, col) as u32),
            ];
            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_write_enable(true)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0);
            let imgui_pipeline = create_graphics_pipeline(
                vulkan_context.device(),
                imgui_render_pass,
                imgui_pipeline_layout,
                vk::PipelineDepthStencilStateCreateInfo::default(),
                vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(binding_descs)
                    .vertex_attribute_descriptions(attribute_descs),
                imgui_vert_module,
                imgui_frag_module,
            )?;
            Either::Right((
                create_graphics_pipeline(
                    vulkan_context.device(),
                    render_pass,
                    pipeline_layout,
                    depth_stencil_state,
                    vk::PipelineVertexInputStateCreateInfo::default(),
                    vert_module,
                    frag_module,
                )?,
                imgui_pipeline,
            ))
        }
    };

    let mut acquire_semaphore = create_semaphore(vulkan_context.device())?;
    let mut present_semaphore = create_semaphore(vulkan_context.device())?;
    let fence = create_fence(vulkan_context.device())?;

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
        &vulkan_context,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &vertices,
    )?;

    let index_buffer = Buffer::from_data(
        &vulkan_context,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::BufferUsageFlags::INDEX_BUFFER,
        &indices,
    )?;
    let mut show_demo = false;
    let start_time = std::time::Instant::now();
    while !handle_window_event(&mut event_pump, &mut imgui, &sdl3_imgui_binding) {
        sdl3_imgui_binding.new_frame(&mut imgui, &window)?;

        check_resize(
            &vulkan_context,
            format,
            render_pass,
            &mut extent,
            &mut swapchain,
            &mut acquire_semaphore,
            &mut present_semaphore,
        )?;

        swapchain.acquire_next_image(acquire_semaphore, vk::Fence::null())?;

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
            vulkan_context.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_barrier],
            );

            let clear_values = vec![
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
                .y(swapchain.extent.height as f32)
                .width(swapchain.extent.width as f32)
                .height(-(swapchain.extent.height as f32))
                .min_depth(0.0)
                .max_depth(1.0)];
            vulkan_context
                .device()
                .cmd_set_viewport(command_buffer, 0, viewports);
            let scissors = &[vk::Rect2D {
                extent: swapchain.extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            }];
            vulkan_context
                .device()
                .cmd_set_scissor(command_buffer, 0, scissors);

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
                    vulkan_context
                        .device()
                        .cmd_begin_rendering(command_buffer, &rendering_info);
                    vert_shader.bind_shader(command_buffer, &[vk::ShaderStageFlags::VERTEX]);
                    frag_shader.bind_shader(command_buffer, &[vk::ShaderStageFlags::FRAGMENT]);
                    Shader::set_vertex_input(command_buffer, &[], &[]);
                    Shader::set_dynamic_state(
                        vulkan_context.device(),
                        command_buffer,
                        viewports,
                        scissors,
                    );
                    push_desc_loader.cmd_push_descriptor_set(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        descriptor_writes,
                    );
                    vulkan_context.device().cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer.buffer,
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
                        indices.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );

                    vulkan_context.device().cmd_end_rendering(command_buffer);
                }
                Either::Right((graphics_pipeline, imgui_pipeline)) => {
                    let render_pass_begin = vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass)
                        .clear_values(&clear_values)
                        .framebuffer(swapchain.framebuffer())
                        .render_area(vk::Rect2D {
                            extent: swapchain.extent,
                            offset: vk::Offset2D { x: 0, y: 0 },
                        });
                    vulkan_context.device().cmd_begin_render_pass(
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

                    vulkan_context.device().cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer.buffer,
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
                    vulkan_context.device().cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *graphics_pipeline,
                    );
                    vulkan_context.device().cmd_draw_indexed(
                        command_buffer,
                        indices.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );

                    vulkan_context.device().cmd_end_render_pass(command_buffer);

                    let clear_values = vec![ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.9, 0.2, 0.1, 1.0],
                        },
                    }];
                    let render_pass_begin = vk::RenderPassBeginInfo::default()
                        .render_pass(imgui_render_pass)
                        .clear_values(&clear_values)
                        .framebuffer(imgui_framebuffers[swapchain.image_index() as usize])
                        .render_area(vk::Rect2D {
                            extent: swapchain.extent,
                            offset: vk::Offset2D { x: 0, y: 0 },
                        });
                    vulkan_context.device().cmd_begin_render_pass(
                        command_buffer,
                        &render_pass_begin,
                        vk::SubpassContents::INLINE,
                    );
                    vulkan_context.device().cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *imgui_pipeline,
                    );

                    let ui = imgui.new_frame();

                    if let Some(wnd) = ui
                        .window("Hello World")
                        .size([300.0, 400.0], imgui::Condition::FirstUseEver)
                        .begin()
                    {
                        if show_demo {
                            ui.show_demo_window(&mut show_demo);
                        }

                        ui.text("Hello, world!");
                        ui.checkbox("Show Demo", &mut show_demo);

                        wnd.end();
                    }

                    let draw_data = imgui.render();
                    render_draw_data(
                        &vulkan_context,
                        draw_data,
                        imgui_pipeline_layout,
                        font_descriptor_set,
                        command_buffer,
                    )?;

                    vulkan_context.device().cmd_end_render_pass(command_buffer);
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

            vulkan_context.device().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present_barrier],
            );

            vulkan_context.device().end_command_buffer(command_buffer)?;

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
            vulkan_context.device().queue_submit(
                vulkan_context.graphics_queue,
                &[submit_info],
                fence,
            )?;

            let swapchains = &[swapchain.swapchain()];
            let present_info = vk::PresentInfoKHR::default()
                .image_indices(images)
                .swapchains(swapchains)
                .wait_semaphores(presents);

            swapchain.queue_present(vulkan_context.graphics_queue, &present_info)?;

            let fences = &[fence];
            vulkan_context
                .device()
                .wait_for_fences(fences, true, u64::MAX)?;
            vulkan_context.device().reset_fences(fences)?;
        };
    }

    unsafe {
        vulkan_context.device().device_wait_idle()?;

        fonts_image.destroy(vulkan_context.device());

        vulkan_context
            .device()
            .free_memory(vertex_buffer.memory, None);
        vulkan_context
            .device()
            .destroy_buffer(vertex_buffer.buffer, None);

        vulkan_context
            .device()
            .free_memory(index_buffer.memory, None);
        vulkan_context
            .device()
            .destroy_buffer(index_buffer.buffer, None);

        vulkan_context.device().destroy_fence(fence, None);
        vulkan_context
            .device()
            .destroy_semaphore(acquire_semaphore, None);
        vulkan_context
            .device()
            .destroy_semaphore(present_semaphore, None);

        vulkan_context
            .device()
            .free_command_buffers(command_pool, &command_buffers);
        vulkan_context
            .device()
            .free_command_buffers(imgui_command_pool, &font_cmds);
        vulkan_context
            .device()
            .destroy_command_pool(command_pool, None);
        vulkan_context
            .device()
            .destroy_command_pool(imgui_command_pool, None);

        // vulkan_context
        //     .device()
        //     .destroy_descriptor_set_layout(desc, None);
        vulkan_context.device().destroy_sampler(font_sampler, None);

        match either_pipeline_or_objects {
            Either::Left((vert_shader, frag_shader)) => {
                vert_shader.destroy();
                frag_shader.destroy();
            }
            Either::Right((graphics_pipeline, imgui_pipeline)) => {
                vulkan_context
                    .device()
                    .destroy_pipeline(graphics_pipeline, None);
                vulkan_context
                    .device()
                    .destroy_pipeline(imgui_pipeline, None);
            }
        }

        vulkan_context
            .device()
            .destroy_descriptor_pool(descriptor_pool, None);
        vulkan_context
            .device()
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        vulkan_context
            .device()
            .destroy_pipeline_layout(pipeline_layout, None);

        swapchain.destroy();
        vulkan_context
            .device()
            .destroy_render_pass(render_pass, None);
    }
    Ok(())
}

fn check_resize(
    vulkan_context: &VulkanContext,
    format: vk::SurfaceFormatKHR,
    render_pass: vk::RenderPass,
    extent: &mut vk::Extent2D,
    swapchain: &mut Swapchain,
    acquire_semaphore: &mut vk::Semaphore,
    present_semaphore: &mut vk::Semaphore,
) -> anyhow::Result<()> {
    let surface_capabilities = vulkan_context
        .surface
        .get_physical_device_surface_capabilities_khr(vulkan_context.physical_device())?;
    if surface_capabilities.current_extent.width != extent.width
        || surface_capabilities.current_extent.height != extent.height
    {
        unsafe {
            vulkan_context.device().device_wait_idle()?;
        }
        *extent = surface_capabilities.current_extent;
        swapchain.recreate(
            vulkan_context.surface(),
            render_pass,
            vulkan_context.physical_device.memory_properties,
            format,
            *extent,
        )?;

        unsafe {
            vulkan_context
                .device()
                .destroy_semaphore(*acquire_semaphore, None);
            vulkan_context
                .device()
                .destroy_semaphore(*present_semaphore, None);

            *acquire_semaphore = create_semaphore(vulkan_context.device())?;
            *present_semaphore = create_semaphore(vulkan_context.device())?;
        }
    };
    Ok(())
}

fn handle_window_event(
    event_pump: &mut sdl3::EventPump,
    imgui: &mut imgui::Context,
    sdl3_imgui_binding: &Sdl3Binding,
) -> bool {
    for event in event_pump.poll_iter() {
        sdl3_imgui_binding.process_event(imgui, &event);
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
