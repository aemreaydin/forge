#version 450

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec4 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} push_constants;

layout(location = 0) out vec2 outTex;

void main() {
    gl_Position = push_constants.mvp * inPos;
    outTex = inTexCoord;
}
