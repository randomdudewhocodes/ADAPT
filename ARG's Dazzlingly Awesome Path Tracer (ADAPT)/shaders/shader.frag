#version 460

layout(binding = 1, rgba32f) uniform readonly image2D storageImage;

ivec2 resolution = imageSize(storageImage);

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 O;

void main()
{
    O = imageLoad(storageImage, ivec2(vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y)));

    O /= O.w;
}