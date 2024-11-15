#version 450

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 tex_coords;
};

layout(binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(location = 0) out vec3 outColor;

void main() {
    Vertex vertex = vertices[gl_VertexIndex];
    outColor = vertex.normal;
    gl_Position = vec4(vertex.position, 1.0);
}
