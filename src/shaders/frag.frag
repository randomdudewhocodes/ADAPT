#version 460

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 O;

void main()
{
    O = vec4(uv, 0, 0);
}