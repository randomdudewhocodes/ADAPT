#version 460

#extension GL_EXT_shader_image_load_formatted : require

layout(binding = 1, rgba32f) uniform readonly image2D storageImage;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 O;

void main()
{
    O = imageLoad(storageImage, ivec2(gl_FragCoord.xy));
}