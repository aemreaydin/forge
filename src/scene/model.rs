use crate::{scene::mesh::Mesh, vulkan_context::VulkanContext};
use nalgebra_glm::Vec3;
use std::sync::Arc;

#[derive(Default, Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
}

#[derive(Default, Debug, Clone)]
pub struct Model {
    #[allow(unused)]
    pub name: String,
    pub meshes: Vec<Arc<Mesh>>,
    pub transform: Transform,

    pub visible: bool,
}

impl Model {
    pub fn new(name: String, meshes: Vec<Arc<Mesh>>, transform: Option<Transform>) -> Self {
        Self {
            name,
            meshes,
            transform: transform.unwrap_or_default(),
            visible: true,
        }
    }

    pub fn destroy(&self, vulkan_context: &VulkanContext) {
        self.meshes.iter().for_each(|mesh| {
            mesh.destroy(vulkan_context);
        });
    }
}
