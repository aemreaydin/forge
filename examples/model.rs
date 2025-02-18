use forge::buffer::Vertex;
use nalgebra_glm::Vec3;

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
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub transform: Transform,
}

impl Model {
    pub fn new(
        name: String,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        transform: Option<Transform>,
    ) -> Self {
        Self {
            name,
            vertices,
            indices,
            transform: transform.unwrap_or_default(),
        }
    }
}
