# SPDX-License-Identifier: Apache-2.0
"""GPU texture networks for the Neural Shading Lab.

A `TextureNet` is one small MLP (optionally with a latent grid) trained on the GPU with
Slang autodiff, following the structure of the course samples in network/step_0x.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import slangpy as spy

HERE = Path(__file__).parent

ENC_RAW, ENC_FREQ, ENC_LATENT = 0, 1, 2
ACT_RELU, ACT_LEAKY, ACT_SINE = 0, 1, 2
ACT_NAMES = {ACT_RELU: "ReLU", ACT_LEAKY: "LeakyReLU", ACT_SINE: "sine"}


@dataclass
class NetConfig:
    enc: int = ENC_FREQ
    octaves: int = 6  # frequency encoding: sin/cos at 6 doubling frequencies
    hidden: int = 32  # 0 = no hidden layer (course step 1)
    act: int = ACT_LEAKY
    out: int = 3
    grid: int = 32  # latent grid resolution
    channels: int = 8  # latent channels
    qbits: int = 0  # 0 = float latents; >0 = quantised latents with noise during training
    lr: float = 0.005
    lr_grid: float = 0.05
    optimizer: str = "adam"  # "adam" or "sgd"
    seed: int = 1
    label: str = ""

    @property
    def inputs(self) -> int:
        if self.enc == ENC_RAW:
            return 2
        if self.enc == ENC_FREQ:
            return 2 + 4 * self.octaves
        return self.channels + 2

    @property
    def latent_channels(self) -> int:
        return self.channels if self.enc == ENC_LATENT else 1

    @property
    def qlevels(self) -> int:
        return (1 << self.qbits) - 1 if (self.enc == ENC_LATENT and self.qbits) else 0

    def type_name(self) -> str:
        if self.hidden == 0:
            return f"NetLinear<{self.enc},{self.inputs},{self.out},{self.latent_channels},{self.qlevels}>"
        return (f"Net<{self.enc},{self.inputs},{self.hidden},{self.out},"
                f"{self.latent_channels},{self.qlevels},{self.act}>")

    def describe(self) -> str:
        sizes = [self.inputs] + ([self.hidden, self.hidden] if self.hidden else []) + [self.out]
        s = " > ".join(str(x) for x in sizes)
        if self.hidden:
            s += f"  {ACT_NAMES[self.act]}"
        return s


_MODULE_SRC = """
import slangpy;
__exported import lab_core;
typealias NetT = {T};

void render(int2 px, int2 res, NetT net, RWTensor<float, 3> result)
{{
    float2 uv = (float2(px) + 0.5f) / float2(res);
    float r[{OUT}] = net.eval(uv, 1, 0);
    [ForceUnroll]
    for (int k = 0; k < {OUT}; ++k)
        result.store(px.y, px.x, k, r[k]);
}}

[Differentiable]
float loss(no_diff float2 uv, NetT net, no_diff float[{OUT}] target, uint seed)
{{
    float r[{OUT}] = net.eval(uv, 0, seed);
    float s = 0;
    [ForceUnroll]
    for (int k = 0; k < {OUT}; ++k)
    {{
        float e = r[k] - target[k];
        s += e * e;
    }}
    return s / {OUT};
}}

void train(uint seed, int2 cell, int2 cells, Tensor<float, 3> reference, NetT net)
{{
    uint h = hash32(seed ^ hash32(uint(cell.x * 7919 + cell.y * 104729)));
    float2 j = float2(float(h & 0xffff), float(h >> 16)) / 65536.0f;
    float2 uv = (float2(cell) + j) / float2(cells);
    int2 res = {{ reference.shape[1], reference.shape[0] }};
    int2 p = min(int2(uv * float2(res)), res - 1);
    uv = (float2(p) + 0.5f) / float2(res);
    float t[{OUT}];
    [ForceUnroll]
    for (int k = 0; k < {OUT}; ++k)
        t[k] = reference.load(p.y, p.x, k);
    bwd_diff(loss)(uv, net, t, h, 1.0f / float(cells.x * cells.y));
}}
"""


class _Params(spy.InstanceList):
    def __init__(self, module, type_name, device, tensors: dict):
        super().__init__(module[type_name])
        self._opt = []
        for name, (arr, grad_name) in tensors.items():
            t = spy.Tensor.from_numpy(device, arr.astype(np.float32))
            g = spy.Tensor.zeros_like(t)
            setattr(self, name, t)
            setattr(self, grad_name, g)
            self._opt.append((t, g, spy.Tensor.zeros_like(t), spy.Tensor.zeros_like(t)))


class TextureNet:
    """One network plus its parameters, optimiser state and training loop."""

    _module_cache: dict = {}

    def __init__(self, device: spy.Device, cfg: NetConfig):
        self.device, self.cfg = device, cfg
        self.module = self._module(device, cfg)
        rng = np.random.default_rng(cfg.seed)
        self.steps = 0
        self.layers = []
        widths = [cfg.inputs] + ([cfg.hidden, cfg.hidden] if cfg.hidden else []) + [cfg.out]
        names = ["l0", "l1", "l2"] if cfg.hidden else ["l2"]
        root = spy.InstanceList(self.module[cfg.type_name()])
        for n, (fi, fo) in zip(names, zip(widths[:-1], widths[1:])):
            last = n == "l2"
            if cfg.act == ACT_SINE and cfg.hidden:
                lim = (1.0 / fi) if n == "l0" else np.sqrt(6.0 / fi) / 30.0
            else:
                lim = np.sqrt(6.0 / fi) * (0.5 if last else 1.0)
            w = rng.uniform(-lim, lim, (fo, fi))
            b = np.full(fo, 0.5) if last else np.zeros(fo)
            layer = _Params(self.module, f"Layer<{fi},{fo}>", device,
                            {"weights": (w, "weights_grad"), "biases": (b, "biases_grad")})
            setattr(root, n, layer)
            self.layers.append(layer)
        g = cfg.grid if cfg.enc == ENC_LATENT else 2
        lat = rng.uniform(0.0, 1.0, (g, g, cfg.latent_channels))
        self.latent = _Params(self.module, f"Latent<{cfg.latent_channels},{cfg.qlevels}>", device,
                              {"texture": (lat, "grads")})
        root.lat = self.latent
        self.root = root
        self.n_weights = sum(fi * fo + fo for fi, fo in zip(widths[:-1], widths[1:]))
        self.n_latents = g * g * cfg.latent_channels if cfg.enc == ENC_LATENT else 0

    @classmethod
    def _module(cls, device, cfg: NetConfig):
        key = (id(device), cfg.type_name())
        if key not in cls._module_cache:
            src = _MODULE_SRC.format(T=cfg.type_name(), OUT=cfg.out)
            name = "labnet_" + "".join(c if c.isalnum() else "_" for c in cfg.type_name())
            cls._module_cache[key] = spy.Module.load_from_source(device, name, src)
        return cls._module_cache[key]

    def train(self, reference: spy.Tensor, iterations: int = 1, batch: int = 64):
        cells = spy.int2(batch, batch)
        for _ in range(iterations):
            self.steps += 1
            self.module.train(seed=(self.steps * 2654435761 + self.cfg.seed) & 0xFFFFFFFF,
                              cell=spy.grid((batch, batch)), cells=cells,
                              reference=reference, net=self.root)
            self._optimise()

    def _optimise(self):
        cfg, m = self.cfg, self.module
        groups = [(l._opt, cfg.lr) for l in self.layers]
        if cfg.enc == ENC_LATENT:
            groups.append((self.latent._opt, cfg.lr_grid))
        for opt, lr in groups:
            for p, g, mm, vv in opt:
                if cfg.optimizer == "sgd":
                    m.sgd_step(p, g, lr)
                else:
                    m.adam_step(p, g, mm, vv, lr, self.steps)
        if cfg.enc == ENC_LATENT and cfg.qlevels:
            m.clamp01(self.latent.texture)

    def render(self, width: int, height: int, out: spy.Tensor | None = None) -> spy.Tensor:
        if out is None or out.shape[0] != height or out.shape[1] != width:
            out = spy.Tensor.empty(self.device, shape=(height, width, self.cfg.out), dtype=float)
        self.module.render(px=spy.grid((height, width)), res=spy.int2(width, height),
                           net=self.root, result=out)
        return out

    def time_inference(self, size: int = 1024, repeats: int = 5) -> float:
        """Milliseconds to evaluate the network for one million pixels."""
        out = spy.Tensor.empty(self.device, shape=(size, size, self.cfg.out), dtype=float)
        self.render(size, size, out)
        self.device.wait()
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.render(size, size, out)
        self.device.wait()
        return (time.perf_counter() - t0) / repeats * 1000.0 * (1_000_000 / (size * size))


def make_device(kind: spy.DeviceType = spy.DeviceType.automatic) -> spy.Device:
    """Device with on-disk shader caches, so only the very first run pays for compilation."""
    cache = HERE / ".cache"
    return spy.Device(type=kind,
                      compiler_options={"include_paths": [spy.SHADER_PATH, str(HERE)]},
                      module_cache_path=str(cache / "modules"),
                      shader_cache_path=str(cache / "shaders"),
                      enable_hot_reload=False, label="neural-shading-lab")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0, 1) - b) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def load_image(path: Path, size: int, crop: float = 1.0) -> np.ndarray:
    """Square crop (fraction `crop` of the short side, top-left), resized, sRGB in [0,1]."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    s = int(min(im.size) * crop)
    im = im.crop((0, 0, s, s)).resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0
