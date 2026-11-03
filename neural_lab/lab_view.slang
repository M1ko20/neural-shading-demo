// SPDX-License-Identifier: Apache-2.0
// Display helpers. The output texture is rgba8_unorm holding sRGB-encoded values, so network
// outputs (trained on sRGB data) are written as they are and the UI draws on top unchanged.
import slangpy;

float3 srgb_to_linear(float3 c)
{
    c = saturate(c);
    return select(c <= 0.04045f, c / 12.92f, pow((c + 0.055f) / 1.055f, 2.4f));
}

float3 linear_to_srgb(float3 c)
{
    c = saturate(c);
    return select(c <= 0.0031308f, c * 12.92f, 1.055f * pow(c, 1.0f / 2.4f) - 0.055f);
}

float3 heat(float t)
{
    // dark teal -> teal -> mint -> orange -> pale yellow (sRGB), for error maps
    const float3 s[5] = { float3(0.07, 0.10, 0.13), float3(0.11, 0.49, 0.45), float3(0.36, 0.81, 0.75),
                          float3(0.91, 0.57, 0.31), float3(0.98, 0.93, 0.63) };
    t = saturate(t) * 4.0f;
    int i = min(int(t), 3);
    return lerp(s[i], s[i + 1], t - float(i));
}

float3 fetch3(Tensor<float, 3> img, int2 p, int ch)
{
    return float3(img.load(p.y, p.x, ch), img.load(p.y, p.x, ch + 1), img.load(p.y, p.x, ch + 2));
}

// mode 0: colour channels ch..ch+2
// mode 1: |img - ref| * gain as a heat map (mean of channels ch..ch+2)
// mode 2: lit preview: albedo in 0..2, tangent-space normal in 3..5, light direction `light`
void show(int2 pixel, int2 size, int2 offset, int mode, int ch, float gain, float3 light,
          Tensor<float, 3> img, Tensor<float, 3> ref, RWTexture2D<float4> output)
{
    int2 shape = { img.shape[1], img.shape[0] };
    int2 p = min(int2((float2(pixel) + 0.5f) / float2(size) * float2(shape)), shape - 1);
    float3 c;
    if (mode == 0)
        c = saturate(fetch3(img, p, ch));
    else if (mode == 1)
    {
        float3 e = abs(saturate(fetch3(img, p, ch)) - fetch3(ref, p, ch));
        c = heat((e.x + e.y + e.z) / 3.0f * gain);
    }
    else
    {
        float3 albedo = srgb_to_linear(fetch3(img, p, 0));
        float3 n = normalize(fetch3(img, p, 3) * 2.0f - 1.0f + float3(0, 0, 1e-4));
        float3 l = normalize(light);
        float3 h = normalize(l + float3(0, 0, 1));
        float diff = saturate(dot(n, l));
        float spec = pow(saturate(dot(n, h)), 40.0f) * 0.25f;
        c = linear_to_srgb(albedo * (0.08f + 1.1f * diff) + spec);
    }
    uint2 dim;
    output.GetDimensions(dim.x, dim.y);
    int2 q = offset + pixel;
    if (all(q >= 0) && all(q < int2(dim)))
        output[q] = float4(c, 1.0f);
}

void clear(int2 pixel, float3 color, RWTexture2D<float4> output)
{
    output[pixel] = float4(color, 1.0f);
}

// Outline rectangle (for captions / frames)
void frame(int2 pixel, int2 size, int2 offset, float3 color, RWTexture2D<float4> output)
{
    if (pixel.x == 0 || pixel.y == 0 || pixel.x == size.x - 1 || pixel.y == size.y - 1)
        output[offset + pixel] = float4(color, 1.0f);
}
