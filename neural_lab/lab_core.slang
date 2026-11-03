// SPDX-License-Identifier: Apache-2.0
// Shared building blocks for the Neural Shading Lab. Same structure as the course's
// network/step_0x samples: differentiable layers whose weight loads carry custom backward
// derivatives that accumulate gradients atomically.
import slangpy;

struct Layer<int In, int Out>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable] float get_bias(int o) { return biases.load(o); }
    [Differentiable] float get_weight(int o, int i) { return weights.load(o, i); }
    [BackwardDerivativeOf(get_bias)] void get_bias_bwd(int o, float g) { biases_grad.add(o, g); }
    [BackwardDerivativeOf(get_weight)] void get_weight_bwd(int o, int i, float g) { weights_grad.add(o, i, g); }

    [Differentiable]
    float[Out] forward(float[In] x)
    {
        float[Out] y;
        // Fully unrolled on purpose. The course samples use [MaxIters(Out)] here, but NVIDIA's
        // Vulkan compiler (NVVM, driver 576) rejects the backward pass of that loop.
        [ForceUnroll]
        for (int r = 0; r < Out; ++r)
        {
            var s = get_bias(r);
            [ForceUnroll]
            for (int c = 0; c < In; ++c)
                s += get_weight(r, c) * x[c];
            y[r] = s;
        }
        return y;
    }
}

uint hash32(uint x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// Learned latent grid (G x G x C). Q > 0: latents live in [0,1] and are stored with Q+1 levels.
// Training adds uniform noise of one quantisation step (as NTC does); rendering rounds.
struct Latent<int C, int Q>
{
    Tensor<float, 3> texture;
    AtomicTensor<float, 3> grads;

    [BackwardDerivative(get_bwd)]
    float get(int x, int y, int c, int mode, uint seed)
    {
        float v = texture.load(y, x, c);
        if (Q > 0)
        {
            v = saturate(v);
            if (mode == 1)
                v = round(v * Q) / Q;
            else
                v += (float(hash32(seed ^ uint(x * 73856093 + y * 19349663 + c * 83492791)) >> 8) * 0x1p-24 - 0.5f) / Q;
        }
        return v;
    }
    void get_bwd(int x, int y, int c, int mode, uint seed, float g) { grads.add(y, x, c, g); }

    [Differentiable]
    float[C] sample(no_diff float2 uv, int mode, uint seed)
    {
        int2 size = { texture.shape[1], texture.shape[0] };
        float2 p = uv * float2(size - 1);
        int2 c0 = clamp(int2(p), int2(0), size - 2);
        float2 f = saturate(p - float2(c0));
        float r[C];
        [ForceUnroll]
        for (int i = 0; i < C; ++i)
        {
            float a = get(c0.x, c0.y, i, mode, seed);
            float b = get(c0.x + 1, c0.y, i, mode, seed);
            float d = get(c0.x, c0.y + 1, i, mode, seed);
            float e = get(c0.x + 1, c0.y + 1, i, mode, seed);
            r[i] = lerp(lerp(a, b, f.x), lerp(d, e, f.x), f.y);
        }
        return r;
    }
}

// Act: 0 = ReLU, 1 = LeakyReLU, 2 = sine (SIREN, w0 = 30)
[Differentiable]
float activate(float x, int act)
{
    if (act == 2) return sin(30.0f * x);
    if (x > 0.0f) return x;
    return act == 1 ? 0.01f * x : 0.0f;
}

// Enc: 0 = raw uv, 1 = uv + sin/cos at (In-2)/4 octaves, 2 = latent grid (C values) + uv
[Differentiable]
float[In] encode<int Enc, int In, int C, int Q>(no_diff float2 uv, Latent<C, Q> lat, int mode, uint seed)
{
    float[In] x;
    [ForceUnroll]
    for (int i = 0; i < In; ++i) x[i] = 0.0f;
    int k = 0;
    if (Enc == 2)
    {
        float l[C] = lat.sample(uv, mode, seed);
        [ForceUnroll]
        for (int i = 0; i < C; ++i) x[i] = l[i];
        k = C;
    }
    x[k] = uv.x * 2 - 1;
    x[k + 1] = uv.y * 2 - 1;
    if (Enc == 1)
    {
        float f = 3.14159265f;
        [ForceUnroll]
        for (int o = 0; o < (In - 2) / 4; ++o)
        {
            x[2 + 4 * o] = sin(f * uv.x);
            x[3 + 4 * o] = cos(f * uv.x);
            x[4 + 4 * o] = sin(f * uv.y);
            x[5 + 4 * o] = cos(f * uv.y);
            f *= 2;
        }
    }
    return x;
}

// Two hidden layers of width H.
struct Net<int Enc, int In, int H, int Out, int C, int Q, int Act>
{
    Latent<C, Q> lat;
    Layer<In, H> l0;
    Layer<H, H> l1;
    Layer<H, Out> l2;

    [Differentiable]
    float[Out] eval(no_diff float2 uv, int mode, uint seed)
    {
        float x[In] = encode<Enc, In, C, Q>(uv, lat, mode, seed);
        float h0[H] = l0.forward(x);
        [ForceUnroll] for (int i = 0; i < H; ++i) h0[i] = activate(h0[i], Act);
        float h1[H] = l1.forward(h0);
        [ForceUnroll] for (int i = 0; i < H; ++i) h1[i] = activate(h1[i], Act);
        return l2.forward(h1);
    }
}

// No hidden layer: the course's step 1.
struct NetLinear<int Enc, int In, int Out, int C, int Q>
{
    Latent<C, Q> lat;
    Layer<In, Out> l2;

    [Differentiable]
    float[Out] eval(no_diff float2 uv, int mode, uint seed)
    {
        float x[In] = encode<Enc, In, C, Q>(uv, lat, mode, seed);
        return l2.forward(x);
    }
}

void adam_step(inout float p, inout float g, inout float m, inout float v, float lr, int t)
{
    m = lerp(g, m, 0.9f);
    v = lerp(g * g, v, 0.999f);
    float mh = m / (1.0f - pow(0.9f, t));
    float vh = v / (1.0f - pow(0.999f, t));
    p -= lr * mh / (sqrt(vh) + 1e-8f);
    g = 0.0f;
}

void sgd_step(inout float p, inout float g, float lr)
{
    p -= lr * g;
    g = 0.0f;
}

void clamp01(inout float p) { p = saturate(p); }
