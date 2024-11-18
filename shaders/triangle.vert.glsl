#version 450

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 tex_coords;
};

layout(binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(push_constant) uniform PushConstants {
    float time;
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
    float xRotation = push_constants.time * 2.0;  // Full rotation every π seconds
    float yRotation = push_constants.time * 1.5;  // Full rotation every 2π/1.5 seconds

    Vertex vertex = vertices[gl_VertexIndex];
    vec3 pos = vertex.position.xyz;
    
    // Apply rotations
    pos = rotateX(pos, xRotation);
    pos = rotateY(pos, yRotation);

    gl_Position = vec4(pos, 1.0);
    float depth = (gl_Position.z + 2.0) / 4.0;
    outColor = vec3(1.0 - depth);
}
