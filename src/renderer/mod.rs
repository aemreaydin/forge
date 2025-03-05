use ::image::ImageReader;
use anyhow::{anyhow, Context};
use ash::vk;
use std::path::Path;

use crate::scene::texture::TextureData;

pub mod base_renderer;
pub mod buffer;
pub mod device;
pub mod image;
pub mod instance;
pub mod physical_device;
pub mod shader_object;
pub mod surface;
pub mod swapchain;
pub mod vulkan_context;

pub enum FrameState {
    PRESENT,
    SUBMIT,
    RESIZED,
}

pub trait Renderer {
    fn start_frame(&self) -> anyhow::Result<()>;
    fn end_frame(&self) -> anyhow::Result<()>;
    fn draw(&mut self) -> anyhow::Result<()>;

    fn resized(&mut self, dims: &[u32; 2]) -> anyhow::Result<bool>;

    fn update(&mut self) -> anyhow::Result<()> {
        self.start_frame()?;
        self.draw()?;
        self.end_frame()
    }
}

pub(crate) fn align_buffer_size(size: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    (size + alignment - 1) & !(alignment - 1)
}

pub fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
) -> anyhow::Result<vk::CommandPool> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(flags);

    Ok(unsafe { device.create_command_pool(&create_info, None)? })
}

pub fn create_descriptor_set_layout(
    device: &ash::Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
    flags: vk::DescriptorSetLayoutCreateFlags,
) -> anyhow::Result<vk::DescriptorSetLayout> {
    let set_create_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(bindings)
        .flags(flags);

    Ok(unsafe { device.create_descriptor_set_layout(&set_create_info, None)? })
}

pub fn create_descriptor_pool(
    device: &ash::Device,
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
    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None)? };
    Ok(descriptor_pool)
}

// TODO: Need a more generalized approach
pub fn create_texture_descriptor_set(
    device: &ash::Device,
    set_layouts: &[vk::DescriptorSetLayout],
    descriptor_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
) -> anyhow::Result<vk::DescriptorSet> {
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(set_layouts);

    unsafe {
        let descriptor_set = device
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
        device.update_descriptor_sets(write_descs, &[]);

        Ok(descriptor_set)
    }
}

pub fn load_shader<P: AsRef<Path>, T: bytemuck::Pod>(path: P) -> anyhow::Result<Vec<T>> {
    let bytes = std::fs::read(path)?;
    Ok(bytemuck::try_cast_slice::<u8, T>(&bytes)
        .expect("Failed to cast shader to u8.")
        .to_vec())
}

pub fn load_image<P: AsRef<Path>>(path: P) -> anyhow::Result<TextureData> {
    let image = ImageReader::open(path)?.decode()?;
    let width = image.width();
    let height = image.height();

    let data = image.to_rgba8().into_vec();

    Ok(TextureData {
        width,
        height,
        data,
    })
}

pub fn create_shader_module<P: AsRef<Path>>(
    device: &ash::Device,
    path: P,
) -> anyhow::Result<vk::ShaderModule> {
    let shader_code = load_shader(path)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
    Ok(unsafe { device.create_shader_module(&create_info, None)? })
}

pub fn create_framebuffer(
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

pub fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    initial_image_layout: vk::ImageLayout,
    has_depth: bool,
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

    if has_depth {
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
    if has_depth {
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

pub fn create_pipeline_layout(
    device: &ash::Device,
    push_constant_ranges: &[vk::PushConstantRange],
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
) -> anyhow::Result<vk::PipelineLayout> {
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(descriptor_set_layouts)
        .push_constant_ranges(push_constant_ranges);
    Ok(unsafe { device.create_pipeline_layout(&create_info, None)? })
}

pub fn create_graphics_pipeline(
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

pub fn create_sampler(device: &ash::Device) -> anyhow::Result<vk::Sampler> {
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
        device
            .create_sampler(&create_info, None)
            .context("Failed to create a sampler")
    }
}

pub fn create_semaphore(device: &ash::Device) -> anyhow::Result<vk::Semaphore> {
    let create_info = vk::SemaphoreCreateInfo::default();
    Ok(unsafe { device.create_semaphore(&create_info, None)? })
}

pub fn create_fence(
    device: &ash::Device,
    flags: vk::FenceCreateFlags,
) -> anyhow::Result<vk::Fence> {
    let create_info = vk::FenceCreateInfo::default().flags(flags);
    Ok(unsafe { device.create_fence(&create_info, None)? })
}
