use forge::buffer::Vertex;
use nalgebra_glm::{Vec2, Vec4};
use tobj::Model;

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
