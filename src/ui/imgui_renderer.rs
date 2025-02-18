use super::imgui_sdl3_platform::ImguiSdlPlatform;
use crate::{
    device::Device,
    image::Image,
    physical_device::PhysicalDevice,
    renderer::{buffer::Buffer, vulkan_context::VulkanContext},
    swapchain::Swapchain,
};
use ash::vk;
use imgui::{
    sys::{ImDrawIdx, ImDrawVert},
    Context, FontAtlasTexture, Ui,
};
use sdl3::{event::Event, EventPump};
use std::sync::Arc;

pub struct ImguiVulkanRenderer {
    device: Arc<Device>,
    physical_device: Arc<PhysicalDevice>,

    sdl_platform: ImguiSdlPlatform,
    imgui_context: Context,

    prev_idx_count: i32,
    prev_vtx_count: i32,

    // Resources
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    framebuffers: Vec<vk::Framebuffer>,
    command_buffers: Vec<vk::CommandBuffer>,
    pub current_command_buffer: vk::CommandBuffer,

    vtx_buffer: Option<Buffer>,
    idx_buffer: Option<Buffer>,
    font_image: Image,
    font_descriptor_set: vk::DescriptorSet,
    font_sampler: vk::Sampler,
}

impl ImguiVulkanRenderer {
    pub fn new(vulkan_context: &VulkanContext) -> anyhow::Result<Self> {
        let mut imgui_context = Context::create();
        let sdl_platform = ImguiSdlPlatform::new(&mut imgui_context);

        let device = vulkan_context.device.clone();
        let physical_device = vulkan_context.physical_device.clone();

        // This is the last render pass to run, so we need to load the previous attachments
        let render_pass = crate::create_render_pass(
            &device.device,
            vulkan_context.surface_format().format,
            vk::AttachmentLoadOp::DONT_CARE,
            vk::ImageLayout::UNDEFINED,
            false,
        )?;

        let descriptor_pool = crate::create_descriptor_pool(
            &device.device,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 100,
            }],
        )?;
        let descriptor_set_layout = crate::create_descriptor_set_layout(
            &device.device,
            &[vk::DescriptorSetLayoutBinding::default()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .binding(0)],
            vk::DescriptorSetLayoutCreateFlags::empty(),
        )?;
        let push_constant_ranges = &[vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            // Scale and Offset for 2D Imgui
            .size(2 * size_of::<nalgebra_glm::Vec2>() as u32)];
        let pipeline_layout = crate::create_pipeline_layout(
            &device.device,
            push_constant_ranges,
            &[descriptor_set_layout],
        )?;

        let vert_module = crate::create_shader_module(&device.device, "shaders/imgui.vert.spv")?;
        let frag_module = crate::create_shader_module(&device.device, "shaders/imgui.frag.spv")?;

        let binding_descs = &[vk::VertexInputBindingDescription::default()
            .stride(size_of::<ImDrawVert>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attribute_descs = &[
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(binding_descs[0].binding)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(ImDrawVert, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(binding_descs[0].binding)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::offset_of!(ImDrawVert, uv) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(binding_descs[0].binding)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(std::mem::offset_of!(ImDrawVert, col) as u32),
        ];

        let pipeline = crate::create_graphics_pipeline(
            &device.device,
            render_pass,
            pipeline_layout,
            vk::PipelineDepthStencilStateCreateInfo::default(),
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(binding_descs)
                .vertex_attribute_descriptions(attribute_descs),
            vert_module,
            frag_module,
        )?;

        let framebuffers = vulkan_context
            .swapchain()
            .image_views()
            .iter()
            .filter_map(|image_view| {
                let views = &[*image_view];
                crate::create_framebuffer(
                    &device.device,
                    render_pass,
                    views,
                    vulkan_context.swapchain_extent().width,
                    vulkan_context.swapchain_extent().height,
                )
                .ok()
            })
            .collect::<Vec<_>>();

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(device.graphics_command_pool);
        let temp_cmd = unsafe { device.device.allocate_command_buffers(&allocate_info)? };

        imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: include_bytes!("../../fonts/imgui_font.ttf"),
                size_pixels: 14.0,
                config: Default::default(),
            }]);
        let fonts = imgui_context.fonts().build_rgba32_texture();
        let font_image = Self::upload_font_image(
            &physical_device,
            &device,
            *vulkan_context.graphics_queue,
            &fonts,
            *temp_cmd
                .first()
                .expect("failed to find a command buffer to use."),
        )?;
        let font_sampler = crate::create_sampler(&device.device)?;
        let font_descriptor_set = crate::create_texture_descriptor_set(
            &device.device,
            &[descriptor_set_layout],
            descriptor_pool,
            font_sampler,
            font_image.image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;
        unsafe {
            device
                .device
                .free_command_buffers(device.graphics_command_pool, &temp_cmd);
        }

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(vulkan_context.swapchain().num_frames_in_flight)
            .command_pool(device.graphics_command_pool);
        let command_buffers = unsafe { device.device.allocate_command_buffers(&allocate_info)? };
        let current_command_buffer = command_buffers[0];

        Ok(Self {
            physical_device,
            device,

            sdl_platform,
            imgui_context,

            prev_idx_count: 0,
            prev_vtx_count: 0,

            vtx_buffer: None,
            idx_buffer: None,

            render_pass,
            framebuffers,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_set_layout,
            command_buffers,
            current_command_buffer,

            font_image,
            font_descriptor_set,
            font_sampler,
        })
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.device.destroy_sampler(self.font_sampler, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .device
                .free_descriptor_sets(self.descriptor_pool, &[self.font_descriptor_set])
                .expect("Failed to free descriptor sets");
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .device
                .destroy_render_pass(self.render_pass, None);
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.font_image.destroy(&self.device.device);
            if let Some(vtx_buffer) = &self.vtx_buffer {
                vtx_buffer.destroy(&self.device.device);
            }
            if let Some(idx_buffer) = &self.idx_buffer {
                idx_buffer.destroy(&self.device.device);
            }
            self.framebuffers.iter().for_each(|framebuffer| {
                self.device.device.destroy_framebuffer(*framebuffer, None);
            });
            self.device
                .device
                .free_command_buffers(self.device.graphics_command_pool, &self.command_buffers);
        }
    }

    pub fn resized(&mut self, swapchain: &Swapchain) -> anyhow::Result<()> {
        unsafe {
            self.framebuffers.iter().for_each(|framebuffer| {
                self.device.device.destroy_framebuffer(*framebuffer, None);
            });

            log::info!(
                "Creating framebuffers with {}-{}",
                swapchain.extent.width,
                swapchain.extent.height
            );
            self.framebuffers = swapchain
                .image_views()
                .iter()
                .filter_map(|image_view| {
                    let views = &[*image_view];
                    crate::create_framebuffer(
                        &self.device.device,
                        self.render_pass,
                        views,
                        swapchain.extent.width,
                        swapchain.extent.height,
                    )
                    .ok()
                })
                .collect::<Vec<_>>();
        }
        Ok(())
    }

    fn upload_font_image(
        physical_device: &PhysicalDevice,
        device: &Device,
        queue: vk::Queue,
        fonts: &FontAtlasTexture,
        cmd: vk::CommandBuffer,
    ) -> anyhow::Result<Image> {
        let image = Image::new(
            physical_device,
            &device.device,
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
        image.copy_to_host(
            physical_device,
            &device.device,
            queue,
            cmd,
            fonts.data,
            vk::Extent3D {
                width: fonts.width,
                height: fonts.height,
                depth: 1,
            },
        )?;
        Ok(image)
    }

    pub fn process_event(&mut self, event: &Event) {
        self.sdl_platform
            .process_event(&mut self.imgui_context, event);
    }

    pub fn start_frame(
        &mut self,
        extent: vk::Extent2D,
        window: &sdl3::video::Window,
        event_pump: &EventPump,
        image_index: usize,
    ) -> anyhow::Result<()> {
        self.sdl_platform
            .new_frame(&mut self.imgui_context, window, event_pump)?;
        self.current_command_buffer = self.command_buffers[image_index];

        let clear_values = &[vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 1.0, 1.0, 1.0],
            },
        }];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .clear_values(clear_values)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                extent,
                offset: vk::Offset2D { x: 0, y: 0 },
            });

        //let to_transfer_barrier = vk::ImageMemoryBarrier::default()
        //    .src_access_mask(vk::AccessFlags::empty())
        //    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
        //    .old_layout(vk::ImageLayout::UNDEFINED)
        //    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        //    .image(self.ctx.swapchain().image(image_index))
        //    .subresource_range(vk::ImageSubresourceRange {
        //        aspect_mask: vk::ImageAspectFlags::COLOR,
        //        base_mip_level: 0,
        //        level_count: 1,
        //        base_array_layer: 0,
        //        layer_count: 1,
        //    })
        //    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        //    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

        unsafe {
            self.device
                .device
                .begin_command_buffer(self.current_command_buffer, &begin_info)?;
            //self.ctx.device().cmd_pipeline_barrier(
            //    self.current_command_buffer,
            //    vk::PipelineStageFlags::TOP_OF_PIPE,
            //    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            //    vk::DependencyFlags::empty(),
            //    &[],
            //    &[],
            //    &[to_transfer_barrier],
            //);
            self.device.device.cmd_begin_render_pass(
                self.current_command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
            self.device.device.cmd_bind_pipeline(
                self.current_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        }

        Ok(())
    }

    pub fn end_frame(&self, image: vk::Image) -> anyhow::Result<()> {
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
            .image(image)
            .subresource_range(subresource_range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        unsafe {
            self.device
                .device
                .cmd_end_render_pass(self.current_command_buffer);
            self.device.device.cmd_pipeline_barrier(
                self.current_command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present_barrier],
            );
            Ok(self
                .device
                .device
                .end_command_buffer(self.current_command_buffer)?)
        }
    }

    pub fn draw(&mut self, mut build: impl FnMut(&mut Ui)) -> anyhow::Result<()> {
        let ui = self.imgui_context.new_frame();
        build(ui);

        let draw_data = self.imgui_context.render();
        let width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        if width <= 0.0
            || height <= 0.0
            || draw_data.total_vtx_count == 0
            || draw_data.total_idx_count == 0
        {
            return Ok(());
        }

        // Most likely same values as the previous frame, no need to recreate buffers

        let vtx_size = crate::align_buffer_size(
            draw_data.total_vtx_count as u64 * size_of::<ImDrawVert>() as u64,
            256,
        );
        let idx_size = crate::align_buffer_size(
            draw_data.total_idx_count as u64 * size_of::<ImDrawIdx>() as u64,
            256,
        );

        let vtx_buffer = Buffer::new(
            &self.physical_device,
            &self.device.device,
            vtx_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let vtx_ptr = vtx_buffer.map_memory(&self.device.device)?;

        let idx_buffer = Buffer::new(
            &self.physical_device,
            &self.device.device,
            idx_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let idx_ptr = idx_buffer.map_memory(&self.device.device)?;

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
            self.device.device.flush_mapped_memory_ranges(ranges)?;
            idx_buffer.unmap_memory(&self.device.device);
            vtx_buffer.unmap_memory(&self.device.device);
        }

        if let Some(idx_buffer) = &self.idx_buffer {
            idx_buffer.destroy(&self.device.device);
        }
        if let Some(vtx_buffer) = &self.vtx_buffer {
            vtx_buffer.destroy(&self.device.device);
        }
        self.prev_idx_count = draw_data.total_idx_count;
        self.prev_vtx_count = draw_data.total_vtx_count;
        self.idx_buffer = Some(idx_buffer);
        self.vtx_buffer = Some(vtx_buffer);

        let viewports = &[vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(width)
            .height(height)
            .min_depth(0.0)
            .max_depth(1.0)];

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

        unsafe {
            self.device
                .device
                .cmd_set_viewport(self.current_command_buffer, 0, viewports);
            self.device.device.cmd_push_constants(
                self.current_command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                scale_bytes,
            );
            self.device.device.cmd_push_constants(
                self.current_command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                (pos.len() * size_of::<f32>()) as u32,
                translate_bytes,
            );
            if let Some(idx_buffer) = &self.idx_buffer {
                self.device.device.cmd_bind_index_buffer(
                    self.current_command_buffer,
                    idx_buffer.buffer,
                    0,
                    if size_of::<ImDrawIdx>() == 2 {
                        vk::IndexType::UINT16
                    } else {
                        vk::IndexType::UINT32
                    },
                );
            }
            if let Some(vtx_buffer) = &self.vtx_buffer {
                self.device.device.cmd_bind_vertex_buffers(
                    self.current_command_buffer,
                    0,
                    &[vtx_buffer.buffer],
                    &[0],
                );
            }

            let clip_off = draw_data.display_pos;
            let clip_scale = draw_data.framebuffer_scale;
            let mut clip_min = [0.0, 0.0];
            let mut clip_max = [0.0, 0.0];
            let mut vtx_offset = 0;
            let mut idx_offset = 0;
            draw_data.draw_lists().for_each(|draw_list| {
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
                            self.device.device.cmd_set_scissor(
                                self.current_command_buffer,
                                0,
                                scissors,
                            );

                            self.device.device.cmd_bind_descriptor_sets(
                                self.current_command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[self.font_descriptor_set],
                                &[],
                            );
                            self.device.device.cmd_draw_indexed(
                                self.current_command_buffer,
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
            self.device
                .device
                .cmd_set_scissor(self.current_command_buffer, 0, scissors);
        }
        Ok(())
    }
}
