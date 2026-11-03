# SPDX-License-Identifier: Apache-2.0
"""Demo 7 support: a scalar volume compressed into a 3D latent grid plus a small shared MLP.

`VolumeNet` follows `TextureNet` in lab_nn.py: Slang autodiff, atomic gradient accumulation,
quantisation noise on the latents during training, Adam. Rendering ray-marches either a dense
voxel grid or the network itself ("inference on sample").
"""
from dataclasses import dataclass
import time

import numpy as np
import slangpy as spy

from lab_nn import _Params


@dataclass
class VolConfig:
    grid: int = 32  # latent grid resolution (G^3)
    channels: int = 8  # latent channels C
    qbits: int = 8  # 0 = float latents
    hidden: int = 32
    lr: float = 0.005
    lr_grid: float = 0.05
    seed: int = 7

    @property
    def inputs(self) -> int:
        return self.channels + 3

    @property
    def qlevels(self) -> int:
        return (1 << self.qbits) - 1 if self.qbits else 0

    def type_name(self) -> str:
        return f"VolNet<{self.inputs},{self.hidden},{self.channels},{self.qlevels}>"

    def latent_bytes(self) -> float:
        return self.grid ** 3 * self.channels * (self.qbits or 32) / 8

    def describe(self) -> str:
        return f"{self.inputs} > {self.hidden} > {self.hidden} > 1  LeakyReLU"


_MODULE_SRC = """
import slangpy;
__exported import lab_volume;
typealias NetT = {T};

struct NeuralField : IField
{{
    NetT net;
    float density(float3 p) {{ return net.eval(p, 1, 0); }}
}}

void render_neural(int2 px, int2 res, float3 eye, float3 fwd, float3 right, float3 up, int steps, float sigma,
                   float3 bg, NetT net, RWTensor<float, 3> result)
{{
    NeuralField f = {{ net }};
    float3 c = march(f, eye, ray_dir(px, res, fwd, right, up, 0.7f), steps, sigma, bg, pixel_jitter(px));
    result.store(px.y, px.x, 0, c.x);
    result.store(px.y, px.x, 1, c.y);
    result.store(px.y, px.x, 2, c.z);
}}

// "Inference on load": evaluate the network once per voxel into a dense grid.
void decode(int3 v, int3 res, NetT net, RWTensor<float, 3> result)
{{
    float3 p = (float3(v) + 0.5f) / float3(res);
    result.store(v.z, v.y, v.x, net.eval(p, 1, 0));
}}

[Differentiable]
float loss(no_diff float3 p, NetT net, no_diff float target, uint seed)
{{
    float e = net.eval(p, 0, seed) - target;
    return e * e;
}}

void train(uint seed, int2 cell, int2 cells, Tensor<float, 3> volume, NetT net)
{{
    uint h = hash32(seed ^ hash32(uint(cell.x * 7919 + cell.y * 104729)));
    uint h1 = hash32(h ^ 0x9e3779b9u), h2 = hash32(h1 ^ 0x85ebca6bu);
    int3 res = {{ volume.shape[2], volume.shape[1], volume.shape[0] }};
    int3 v = int3(int(h1 % uint(res.x)), int(h2 % uint(res.y)), int(hash32(h2) % uint(res.z)));
    float3 p = (float3(v) + 0.5f) / float3(res);
    float t = volume.load(v.z, v.y, v.x);
    bwd_diff(loss)(p, net, t, h, 1.0f / float(cells.x * cells.y));
}}
"""


class VolumeNet:
    _module_cache: dict = {}

    def __init__(self, device: spy.Device, cfg: VolConfig):
        self.device, self.cfg = device, cfg
        self.module = self._module(device, cfg)
        rng = np.random.default_rng(cfg.seed)
        self.steps = 0
        root = spy.InstanceList(self.module[cfg.type_name()])
        widths = [cfg.inputs, cfg.hidden, cfg.hidden, 1]
        self.layers = []
        for n, (fi, fo) in zip(["l0", "l1", "l2"], zip(widths[:-1], widths[1:])):
            last = n == "l2"
            lim = np.sqrt(6.0 / fi) * (0.5 if last else 1.0)
            layer = _Params(self.module, f"Layer<{fi},{fo}>", device,
                            {"weights": (rng.uniform(-lim, lim, (fo, fi)), "weights_grad"),
                             "biases": (np.zeros(fo), "biases_grad")})
            setattr(root, n, layer)
            self.layers.append(layer)
        g = cfg.grid
        lat = rng.uniform(0.0, 1.0, (g, g, g, cfg.channels))
        self.latent = _Params(self.module, f"Latent3D<{cfg.channels},{cfg.qlevels}>", device,
                              {"texture": (lat, "grads")})
        root.lat = self.latent
        self.root = root
        self.n_weights = sum(fi * fo + fo for fi, fo in zip(widths[:-1], widths[1:]))
        self.n_latents = g ** 3 * cfg.channels

    @classmethod
    def _module(cls, device, cfg: VolConfig):
        key = (id(device), cfg.type_name())
        if key not in cls._module_cache:
            name = "labvol_" + "".join(c if c.isalnum() else "_" for c in cfg.type_name())
            cls._module_cache[key] = spy.Module.load_from_source(device, name, _MODULE_SRC.format(T=cfg.type_name()))
        return cls._module_cache[key]

    def train(self, volume: spy.Tensor, iterations: int = 1, batch: int = 64):
        m, cfg = self.module, self.cfg
        for _ in range(iterations):
            self.steps += 1
            m.train(seed=(self.steps * 2654435761 + cfg.seed) & 0xFFFFFFFF, cell=spy.grid((batch, batch)),
                    cells=spy.int2(batch, batch), volume=volume, net=self.root)
            for opt, lr in [(l._opt, cfg.lr) for l in self.layers] + [(self.latent._opt, cfg.lr_grid)]:
                for p, g, mm, vv in opt:
                    m.adam_step(p, g, mm, vv, lr, self.steps)
            if cfg.qlevels:
                m.clamp01(self.latent.texture)

    def decode(self, n: int, out: spy.Tensor | None = None) -> spy.Tensor:
        if out is None or tuple(out.shape) != (n, n, n):
            out = spy.Tensor.empty(self.device, shape=(n, n, n), dtype=float)
        self.module.decode(v=spy.grid((n, n, n)), res=spy.int3(n, n, n), net=self.root, result=out)
        return out

    def render(self, cam: dict, size: int, out: spy.Tensor | None = None) -> spy.Tensor:
        if out is None or tuple(out.shape) != (size, size, 3):
            out = spy.Tensor.empty(self.device, shape=(size, size, 3), dtype=float)
        self.module.render_neural(px=spy.grid((size, size)), res=spy.int2(size, size), net=self.root,
                                  result=out, **cam)
        return out


def render_dense(module, vol: spy.Tensor, cam: dict, size: int, out: spy.Tensor | None = None) -> spy.Tensor:
    if out is None or tuple(out.shape) != (size, size, 3):
        out = spy.Tensor.empty(module.device, shape=(size, size, 3), dtype=float)
    module.render_dense(px=spy.grid((size, size)), res=spy.int2(size, size), vol=vol, result=out, **cam)
    return out


def camera(angle: float, elev: float = 0.38, dist: float = 1.6, steps: int = 160, sigma: float = 28.0,
           bg=(17 / 255, 26 / 255, 34 / 255)) -> dict:
    c = np.array([0.5, 0.5, 0.5])
    eye = c + dist * np.array([np.cos(angle) * np.cos(elev), np.sin(elev), np.sin(angle) * np.cos(elev)])
    fwd = c - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f3 = lambda v: spy.float3(*[float(x) for x in v])
    return dict(eye=f3(eye), fwd=f3(fwd), right=f3(right), up=f3(up), steps=steps, sigma=sigma, bg=f3(bg))


def time_ms(fn, repeats: int = 3, device=None) -> float:
    fn()
    device.wait()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    device.wait()
    return (time.perf_counter() - t0) / repeats * 1000.0


# ---------------------------------------------------------------------------------------------
# Data: a procedural "simulation-like" density field and a same-bytes dense baseline
# ---------------------------------------------------------------------------------------------
def make_volume(n: int, seed: int = 3) -> np.ndarray:
    """Clumpy medium with filaments and sharp-edged shells, values in [0,1], shape (z, y, x).

    Stands in for astrophysical / SPH density: smooth clumps (easy for any method) plus thin
    filaments and shock-like shells (where a downsampled grid blurs and a decoder can stay sharp).
    """
    rng = np.random.default_rng(seed)
    ax = (np.arange(n, dtype=np.float32) + 0.5) / n
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
    p = np.stack([x, y, z], -1)
    d = np.zeros((n, n, n), np.float32)
    for _ in range(18):  # clumps
        c = rng.uniform(0.18, 0.82, 3)
        s = rng.uniform(0.018, 0.055)
        d += rng.uniform(0.3, 0.9) * np.exp(-np.sum((p - c) ** 2, -1) / (2 * s * s))
    for _ in range(7):  # filaments: thin tubes along random curved segments
        a, b = rng.uniform(0.15, 0.85, 3), rng.uniform(0.15, 0.85, 3)
        bend = rng.normal(0, 0.12, 3)
        r = rng.uniform(0.006, 0.014)
        best = np.full((n, n, n), 1e9, np.float32)
        for t in np.linspace(0, 1, 48):
            q = a + (b - a) * t + bend * 4 * t * (1 - t)
            best = np.minimum(best, np.sum((p - q) ** 2, -1))
        d += 1.2 * np.exp(-best / (2 * r * r))
    for _ in range(2):  # shock shells: sharp outer edge, soft inner falloff, rippled
        c = rng.uniform(0.35, 0.65, 3)
        R = rng.uniform(0.14, 0.22)
        v = p - c
        rr = np.sqrt(np.sum(v * v, -1))
        ripple = 0.012 * np.sin(14 * v[..., 0] / R) * np.sin(11 * v[..., 1] / R)
        e = rr - (R + ripple)
        shell = np.where(e < 0, np.exp(e / 0.03), np.exp(-e / 0.004))
        d += 0.6 * shell * (rr < R + 0.06)
    d = 1.0 - np.exp(-1.4 * d)  # soft tone-map into [0,1]
    return np.clip(d, 0, 1).astype(np.float32)


def resize3(v: np.ndarray, m: int) -> np.ndarray:
    """Separable linear resample of a cube volume to m^3 (cell-centred), area-averaged when shrinking."""
    out = v
    for axis in range(3):
        n = out.shape[axis]
        if m < n:  # box pre-filter so downsampling averages instead of aliasing
            k = n / m
            src = (np.arange(m) + 0.5) * k - 0.5
            lo = np.floor(src - k / 2 + 0.5).astype(int)
            hi = np.floor(src + k / 2 + 0.5).astype(int)
            cs = np.cumsum(np.moveaxis(out, axis, 0), 0)
            cs = np.concatenate([np.zeros((1,) + cs.shape[1:], cs.dtype), cs], 0)
            lo, hi = np.clip(lo, 0, n), np.clip(hi, 1, n)
            res = (cs[hi] - cs[lo]) / (hi - lo).reshape((-1,) + (1,) * (out.ndim - 1))
            out = np.moveaxis(res, 0, axis)
        else:
            src = (np.arange(m) + 0.5) * n / m - 0.5
            i0 = np.clip(np.floor(src).astype(int), 0, n - 1)
            i1 = np.clip(i0 + 1, 0, n - 1)
            f = np.clip(src - i0, 0, 1).reshape((-1,) + (1,) * (out.ndim - 1))
            a = np.moveaxis(out, axis, 0)
            out = np.moveaxis(a[i0] * (1 - f) + a[i1] * f, 0, axis)
    return out.astype(np.float32)


def volume_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0, 1) - b) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))
