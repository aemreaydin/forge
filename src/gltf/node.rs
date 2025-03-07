use nalgebra_glm::Mat4;

use crate::{camera::Camera, scene::mesh::Mesh};

pub enum NodeType {
    Root,
    Mesh(Mesh),
    Camera(Camera),
}

pub struct Node {
    children: Option<Vec<Node>>,
    transform: Option<Mat4>,
    node: NodeType,
}

impl Node {
    //pub fn new(node: &gltf::Node) -> Self {
    //    let node = if let Some(camera) = node.camera() {
    //        match camera.projection() {
    //            gltf::camera::Projection::Orthographic(orthographic) => todo!(),
    //            gltf::camera::Projection::Perspective(perspective) => NodeType::Camera(Camera::new()),
    //        }
    //    } else if let Some(mesh) = node.mesh {
    //    } else {
    //    };
    //    //let children = node.children().map(|child| {
    //    //    child.
    //    //
    //    //});
    //}
}
