#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tex_coords;

layout(location = 0) out vec3 outColor;

void main() {
    outColor = normal;
    gl_Position = vec4(position, 1.0);
}
