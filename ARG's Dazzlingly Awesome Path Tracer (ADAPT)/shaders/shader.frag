#version 460

layout(location = 0) in vec2 I;

layout(location = 0) out vec4 O;

void main()
{
    O = vec4(I, 0, 1);
}