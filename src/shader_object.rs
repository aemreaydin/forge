use anyhow::anyhow;
use ash::{
    ext,
    vk::{self, ColorComponentFlags},
};
use std::{ffi::CStr, sync::OnceLock};

static SHADER_OBJECT_DEVICE_FNS: OnceLock<ext::shader_object::Device> = OnceLock::new();

pub struct Shader {
    pub shader: vk::ShaderEXT,
}

impl Shader {
    // TODO: For now we will only support unlinked shaders
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        name: &CStr,
        stage: vk::ShaderStageFlags,
        code: &[u8],
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> anyhow::Result<Self> {
        let loader = SHADER_OBJECT_DEVICE_FNS
            .get_or_init(|| ext::shader_object::Device::new(instance, device));

        let shader = unsafe {
            let create_infos = &[vk::ShaderCreateInfoEXT::default()
                .stage(stage)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(code)
                .name(name)
                .set_layouts(set_layouts)
                .push_constant_ranges(push_constant_ranges)];
            match loader.create_shaders(create_infos, None) {
                Ok(shaders) => shaders[0], // TODO: There should be at least one shader
                Err((_, vk_result)) => Err(anyhow!(
                    "failed to create shader with error {}",
                    vk_result.to_string()
                ))?,
            }
        };
        Ok(Self { shader })
    }

    pub fn bind_shader(&self, command_buffer: vk::CommandBuffer, stages: &[vk::ShaderStageFlags]) {
        unsafe { Self::get_device_fns().cmd_bind_shaders(command_buffer, stages, &[self.shader]) }
    }

    pub fn set_dynamic_state(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        viewports: &[vk::Viewport],
        scissors: &[vk::Rect2D],
    ) {
        unsafe {
            let device_fns = Self::get_device_fns();
            device_fns.cmd_set_primitive_restart_enable(command_buffer, false);
            device_fns.cmd_set_alpha_to_coverage_enable(command_buffer, false);
            device_fns.cmd_set_viewport_with_count(command_buffer, viewports);
            device_fns.cmd_set_scissor_with_count(command_buffer, scissors);
            device_fns.cmd_set_rasterization_samples(command_buffer, vk::SampleCountFlags::TYPE_1);
            device_fns.cmd_set_sample_mask(command_buffer, vk::SampleCountFlags::TYPE_1, &[0xFF]);
            device_fns.cmd_set_polygon_mode(command_buffer, vk::PolygonMode::FILL);
            device_fns.cmd_set_rasterizer_discard_enable(command_buffer, false);
            device_fns.cmd_set_color_write_mask(command_buffer, 0, &[ColorComponentFlags::RGBA]);
            device_fns.cmd_set_depth_test_enable(command_buffer, true);
            device_fns.cmd_set_depth_write_enable(command_buffer, true);
            device_fns.cmd_set_depth_compare_op(command_buffer, vk::CompareOp::LESS);
            device_fns.cmd_set_depth_bounds_test_enable(command_buffer, true);
            device.cmd_set_depth_bounds(command_buffer, 0.0, 1.0);
            device_fns.cmd_set_depth_bias_enable(command_buffer, true);
            device.cmd_set_depth_bias(command_buffer, 0.0, 0.0, 0.0);
            device_fns.cmd_set_stencil_test_enable(command_buffer, false);
            device_fns.cmd_set_cull_mode(command_buffer, vk::CullModeFlags::BACK);
            device_fns.cmd_set_front_face(command_buffer, vk::FrontFace::COUNTER_CLOCKWISE);
            device_fns.cmd_set_color_blend_enable(command_buffer, 0, &[vk::TRUE]);
            device_fns.cmd_set_color_blend_equation(
                command_buffer,
                0,
                &[vk::ColorBlendEquationEXT::default()
                    .src_color_blend_factor(vk::BlendFactor::ONE)
                    .dst_color_blend_factor(vk::BlendFactor::ZERO)
                    .color_blend_op(vk::BlendOp::ADD)],
            );
            device_fns
                .cmd_set_primitive_topology(command_buffer, vk::PrimitiveTopology::TRIANGLE_LIST);
        }
    }

    pub fn set_vertex_input(
        command_buffer: vk::CommandBuffer,
        vertex_input_descs: &[vk::VertexInputBindingDescription2EXT<'_>],
        vertex_atturibute_descs: &[vk::VertexInputAttributeDescription2EXT<'_>],
    ) {
        unsafe {
            Self::get_device_fns().cmd_set_vertex_input(
                command_buffer,
                vertex_input_descs,
                vertex_atturibute_descs,
            )
        }
    }

    pub fn destroy(&self) {
        unsafe { Self::get_device_fns().destroy_shader(self.shader, None) }
    }

    fn get_device_fns() -> &'static ext::shader_object::Device {
        SHADER_OBJECT_DEVICE_FNS
            .get()
            .expect("shader object device functions not initialized")
    }
}
