#version 450

struct Vertex {
    vec4 position;
    vec4 normal;
    vec3 tex_coords;
};

layout(binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(push_constant) uniform PushConstants {
  mat4 mvp;
} push_constants;

layout(location = 0) out vec3 outColor;

vec3 rotateX(vec3 pos, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec3(
        pos.x,
        pos.y * c - pos.z * s,
        pos.y * s + pos.z * c
    );
}

vec3 rotateY(vec3 pos, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec3(
        pos.x * c + pos.z * s,
        pos.y,
        -pos.x * s + pos.z * c
    );
}

void main() {
    // float xRotation = push_constants.time * 2.0;  // Full rotation every π seconds
    // float yRotation = push_constants.time * 1.5;  // Full rotation every 2π/1.5 seconds

    Vertex vertex = vertices[gl_VertexIndex];
    // vec3 pos = vertex.position.xyz;
    
    gl_Position = push_constants.mvp * vertex.position;
    outColor = vec3((vertex.normal * 0.5).xyz + 0.5);
    // Apply rotations
    // pos = rotateX(pos, xRotation);
    // pos = rotateY(pos, yRotation);

    // gl_Position = vec4(pos, 1.0);
}
