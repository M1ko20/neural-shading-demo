// SPDX-License-Identifier: Apache-2.0
// Demo 7: a scalar volume as a 3D latent grid plus a small shared MLP, ray-marched live.
// Same building blocks as the 2D demos (lab_core.slang): differentiable layers with atomic
// gradient accumulation, a quantised latent grid trained with noise, Adam.
import slangpy;
__exported import lab_core;

// Learned latent grid G x G x G x C (z, y, x, c). Q > 0: stored with Q+1 levels, as in Latent.
struct Latent3D<int C, int Q>
{
    Tensor<float, 4> texture;
    AtomicTensor<float, 4> grads;

    [BackwardDerivative(get_bwd)]
    float get(int3 p, int c, int mode, uint seed)
    {
        float v = texture.load(p.z, p.y, p.x, c);
        if (Q > 0)
        {
            v = saturate(v);
            if (mode == 1)
                v = round(v * Q) / Q;
            else
                v += (float(hash32(seed ^ uint(p.x * 73856093 + p.y * 19349663 + p.z * 50331653 + c * 83492791)) >> 8)
                      * 0x1p-24 - 0.5f) / Q;
        }
        return v;
    }
    void get_bwd(int3 p, int c, int mode, uint seed, float g) { grads.add(p.z, p.y, p.x, c, g); }

    [Differentiable]
    float[C] sample(no_diff float3 uvw, int mode, uint seed)
    {
        int3 size = { texture.shape[2], texture.shape[1], texture.shape[0] };
        float3 q = saturate(uvw) * float3(size - 1);
        int3 c0 = clamp(int3(q), int3(0), size - 2);
        float3 f = saturate(q - float3(c0));
        float r[C];
        [ForceUnroll]
        for (int i = 0; i < C; ++i)
        {
            float v000 = get(c0 + int3(0, 0, 0), i, mode, seed);
            float v100 = get(c0 + int3(1, 0, 0), i, mode, seed);
            float v010 = get(c0 + int3(0, 1, 0), i, mode, seed);
            float v110 = get(c0 + int3(1, 1, 0), i, mode, seed);
            float v001 = get(c0 + int3(0, 0, 1), i, mode, seed);
            float v101 = get(c0 + int3(1, 0, 1), i, mode, seed);
            float v011 = get(c0 + int3(0, 1, 1), i, mode, seed);
            float v111 = get(c0 + int3(1, 1, 1), i, mode, seed);
            float a = lerp(lerp(v000, v100, f.x), lerp(v010, v110, f.x), f.y);
            float b = lerp(lerp(v001, v101, f.x), lerp(v011, v111, f.x), f.y);
            r[i] = lerp(a, b, f.z);
        }
        return r;
    }
}

// Latent features (C) + position (3) -> H -> H -> 1 density. In = C + 3. LeakyReLU.
struct VolNet<int In, int H, int C, int Q>
{
    Latent3D<C, Q> lat;
    Layer<In, H> l0;
    Layer<H, H> l1;
    Layer<H, 1> l2;

    [Differentiable]
    float eval(no_diff float3 p, int mode, uint seed)
    {
        float l[C] = lat.sample(p, mode, seed);
        float x[In];
        [ForceUnroll]
        for (int i = 0; i < C; ++i) x[i] = l[i];
        x[C] = p.x * 2 - 1;
        x[C + 1] = p.y * 2 - 1;
        x[C + 2] = p.z * 2 - 1;
        float h0[H] = l0.forward(x);
        [ForceUnroll] for (int i = 0; i < H; ++i) h0[i] = activate(h0[i], 1);
        float h1[H] = l1.forward(h0);
        [ForceUnroll] for (int i = 0; i < H; ++i) h1[i] = activate(h1[i], 1);
        return l2.forward(h1)[0];
    }
}

// ---------------------------------------------------------------------------------------------
// Ray marching: one front-to-back emission/absorption integrator for both field kinds
// ---------------------------------------------------------------------------------------------
interface IField
{
    float density(float3 p);
}

// Dense voxel grid (z, y, x), trilinear like a 3D texture.
struct DenseField : IField
{
    Tensor<float, 3> vol;
    float density(float3 p)
    {
        int3 size = { vol.shape[2], vol.shape[1], vol.shape[0] };
        float3 q = saturate(p) * float3(size) - 0.5f;
        int3 c0 = clamp(int3(floor(q)), int3(0), size - 2);
        float3 f = saturate(q - float3(c0));
        float v000 = vol.load(c0.z, c0.y, c0.x), v100 = vol.load(c0.z, c0.y, c0.x + 1);
        float v010 = vol.load(c0.z, c0.y + 1, c0.x), v110 = vol.load(c0.z, c0.y + 1, c0.x + 1);
        float v001 = vol.load(c0.z + 1, c0.y, c0.x), v101 = vol.load(c0.z + 1, c0.y, c0.x + 1);
        float v011 = vol.load(c0.z + 1, c0.y + 1, c0.x), v111 = vol.load(c0.z + 1, c0.y + 1, c0.x + 1);
        float a = lerp(lerp(v000, v100, f.x), lerp(v010, v110, f.x), f.y);
        float b = lerp(lerp(v001, v101, f.x), lerp(v011, v111, f.x), f.y);
        return lerp(a, b, f.z);
    }
}

float3 tf_colour(float d)
{
    // deck palette: teal -> mint -> orange -> pale yellow, in linear-ish sRGB
    const float3 s[4] = { float3(0.11, 0.49, 0.45), float3(0.36, 0.81, 0.75),
                          float3(0.91, 0.57, 0.31), float3(0.98, 0.93, 0.63) };
    float t = saturate(d) * 3.0f;
    int i = min(int(t), 2);
    return lerp(s[i], s[i + 1], t - float(i));
}

float3 march<F : IField>(F field, float3 eye, float3 dir, int steps, float sigma, float3 bg, float jitter)
{
    // ray vs unit cube
    float3 inv = 1.0f / dir;
    float3 t0 = (float3(0) - eye) * inv, t1 = (float3(1) - eye) * inv;
    float tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    float tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (tf <= max(tn, 0.0f))
        return bg;
    tn = max(tn, 0.0f);
    float dt = 1.7320508f / float(steps);
    float3 acc = float3(0);
    float trans = 1.0f;
    for (float t = tn + jitter * dt; t < tf; t += dt)
    {
        float d = saturate(field.density(eye + t * dir));
        float w = smoothstep(0.06f, 0.6f, d);  // transfer function: thin media stay transparent
        if (w > 0.0f)
        {
            float a = 1.0f - exp(-sigma * w * dt);
            acc += trans * a * tf_colour(d);
            trans *= 1.0f - a;
            if (trans < 0.01f)
                break;
        }
    }
    return acc + trans * bg;
}

// Per-pixel start offset (fixed, so every pane gets the same noise): removes step banding.
float pixel_jitter(int2 px) { return float(hash32(uint(px.x * 9781 + px.y * 6271)) >> 8) * 0x1p-24; }

float3 ray_dir(int2 px, int2 res, float3 fwd, float3 right, float3 up, float fov)
{
    float2 ndc = (float2(px) + 0.5f) / float2(res) * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float s = tan(0.5f * fov);
    return normalize(fwd + ndc.x * s * right + ndc.y * s * up);
}

void render_dense(int2 px, int2 res, float3 eye, float3 fwd, float3 right, float3 up, int steps, float sigma,
                  float3 bg, Tensor<float, 3> vol, RWTensor<float, 3> result)
{
    DenseField f = { vol };
    float3 c = march(f, eye, ray_dir(px, res, fwd, right, up, 0.7f), steps, sigma, bg, pixel_jitter(px));
    result.store(px.y, px.x, 0, c.x);
    result.store(px.y, px.x, 1, c.y);
    result.store(px.y, px.x, 2, c.z);
}
