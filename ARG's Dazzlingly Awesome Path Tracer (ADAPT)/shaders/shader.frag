#version 460

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 O;

void main()
{
    O = texture(texSampler, uv);
}