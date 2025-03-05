#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 texCoords;
layout(set = 0, binding = 1) uniform sampler2D tex;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = texture(tex, texCoords);
}
