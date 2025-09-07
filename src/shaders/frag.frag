#version 460

layout(binding = 0, rgba32f) uniform readonly image2D resultImage;

ivec2 resolution = imageSize(resultImage);

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 O;

void main()
{
    O = imageLoad(resultImage, ivec2(vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y)));

    O /= O.w;
}