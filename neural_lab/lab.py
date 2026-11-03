# SPDX-License-Identifier: Apache-2.0
"""Neural Shading Lab: live GPU demos for the talk "Neural Shading & Neural Compression".

Run from the repository's virtual environment:

    python neural_lab/lab.py              # open the lab window
    python neural_lab/lab.py --demo 4     # start on a specific demo (1-7)
    python neural_lab/lab.py --precompile # compile every shader first (do this before the talk)

Keys: 1-7 switch demo, Space pause/resume, R restart, Left/Right previous/next preset,
F2 screenshot (screenshot.png), Esc quit.
"""
import argparse
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import slangpy as spy

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from lab_nn import (ACT_LEAKY, ACT_RELU, ACT_SINE, ENC_FREQ, ENC_LATENT, ENC_RAW,  # noqa: E402
                    NetConfig, TextureNet, load_image, make_device, psnr)
import bcn  # noqa: E402
from lab_volume import (VolConfig, VolumeNet, camera, make_volume, render_dense,  # noqa: E402
                        resize3, time_ms, volume_psnr)

REPO = HERE.parent
IMAGES = {
    "slangstars.png": lambda s: load_image(REPO / "network" / "slangstars.png", s),
    "PavingStones albedo": lambda s: load_image(REPO / "mipmap" / "PavingStones070_2K.diffuse.jpg", s, 0.5),
    "PavingStones normal": lambda s: load_image(REPO / "mipmap" / "PavingStones070_2K.normal.jpg", s, 0.5),
}

WIDTH, HEIGHT = 1600, 900
PANEL_W = 400
VIEW_X = PANEL_W + 24
VIEW_W = WIDTH - VIEW_X - 24
CAPTION_H = 52
BG = (17 / 255, 26 / 255, 34 / 255)  # #111A22, the deck's dark colour
TEAL = (28, 124, 114)
COLORS = [(92, 207, 191), (232, 145, 79), (150, 170, 240), (240, 220, 120), (200, 120, 200)]


def _v(x, default):
    return default if x is None else x


def wrap(text: str, width: int = 50) -> str:
    return "\n".join(textwrap.fill(p, width) if p else "" for p in text.split("\n"))


# --------------------------------------------------------------------------------------------
# Small NumPy chart renderer (curves as an image, drawn into the GPU view like any other pane)
# --------------------------------------------------------------------------------------------
def chart(series, w, h, ymin=None, ymax=None, hline=None):
    img = np.empty((h, w, 3), np.float32)
    img[:] = np.array([23, 35, 45]) / 255
    pts_all = [p for s, _ in series for p in s]
    if not pts_all:
        return img, (0, 1)
    ys = [p[1] for p in pts_all] + ([hline[0]] if hline else [])
    lo = ymin if ymin is not None else min(ys) - 1
    hi = ymax if ymax is not None else max(ys) + 1
    xmax = max(p[0] for p in pts_all) or 1
    pad = 12
    def X(x): return pad + x / xmax * (w - 2 * pad)
    def Y(y): return h - pad - (y - lo) / (hi - lo + 1e-9) * (h - 2 * pad)
    for k in range(5):  # grid
        yy = int(round(Y(lo + (hi - lo) * k / 4)))
        if 0 <= yy < h:
            img[yy, pad:w - pad] = np.array([42, 58, 68]) / 255
    def line(pts, col, thick=2, dashed=False):
        col = np.array(col) / 255
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = int(max(abs(X(x1) - X(x0)), abs(Y(y1) - Y(y0)))) + 1
            for t in np.linspace(0, 1, n):
                if dashed and int(X(x0) + t * (X(x1) - X(x0))) // 8 % 2:
                    continue
                cx, cy = X(x0) + t * (X(x1) - X(x0)), Y(y0) + t * (Y(y1) - Y(y0))
                ix, iy = int(cx), int(cy)
                img[max(0, iy - thick // 2):iy + (thick + 1) // 2, max(0, ix - thick // 2):ix + (thick + 1) // 2] = col
    if hline:
        line([(0, hline[0]), (xmax, hline[0])], hline[1], 2, True)
    for pts, col in series:
        if len(pts) > 1:
            line(pts, col, 3)
    return img, (lo, hi)


# --------------------------------------------------------------------------------------------
# Demos
# --------------------------------------------------------------------------------------------
class Demo:
    title = ""
    part = ""
    deck = ""  # which slides this supports
    blurb = ""

    def __init__(self, lab: "Lab"):
        self.lab, self.device = lab, lab.device
        self.paused = False

    def build_ui(self, parent): ...
    def reset(self): ...
    def update(self): ...
    def panes(self) -> list: return []  # [(caption, subtitle, x, y, w, h, drawfn)]
    def stats(self) -> str: return ""
    def preset(self, delta: int): ...
    def warm(self): self.reset()


class FitDemo(Demo):
    title = "Fit a texture"
    part = "Part 1 · The idea"
    deck = "The simplest texture network · Frequency encoding · Adding latent textures"
    blurb = ("A texture is a function from uv to colour. A small MLP learns it by gradient descent "
             "on random pixels. The presets are the five steps of network/ in the course repo, "
             "plus SIREN.")
    PRESETS = [
        ("1  One layer (step_01)", NetConfig(enc=ENC_RAW, hidden=0, lr=0.02),
         "2 inputs, 3 outputs: 9 parameters. All it can learn is a linear gradient."),
        ("2  Hidden layers, ReLU (step_02)", NetConfig(enc=ENC_RAW, act=ACT_RELU),
         "Two hidden layers of 32. ReLU is piecewise linear: look for straight edges and triangles."),
        ("3  LeakyReLU (step_03)", NetConfig(enc=ENC_RAW, act=ACT_LEAKY),
         "Fewer dead neurons, still blurry: the problem is the input, not the activation."),
        ("4  Frequency encoding (step_04)", NetConfig(enc=ENC_FREQ, act=ACT_LEAKY),
         "Same network, input is sin/cos of uv at 6 doubling frequencies. Spectral bias is worked "
         "around and fine detail appears."),
        ("5  Latent grid (step_05)", NetConfig(enc=ENC_LATENT, grid=32, channels=8, act=ACT_LEAKY),
         "A coarse grid of learned 8-number codes feeds a tiny decoder. Detail lives in the grid, "
         "not in the weights."),
        ("SIREN  sine activations", NetConfig(enc=ENC_RAW, act=ACT_SINE, lr=0.0006),
         "Sine activations give a raw-uv network high frequencies with no encoding. Same family "
         "as the SIREN octree in our volume work."),
    ]

    def __init__(self, lab):
        super().__init__(lab)
        self.idx, self.target_name, self.iters, self.batch = 3, "slangstars.png", 10, 64
        self.res = 128
        self.net = None

    def build_ui(self, p):
        spy.ui.ComboBox(p, "Preset", self.idx, lambda v: self._set(v), [n for n, _, _ in self.PRESETS])
        spy.ui.ComboBox(p, "Target", list(IMAGES).index(self.target_name),
                        lambda v: self._target(list(IMAGES)[v]), list(IMAGES))
        spy.ui.SliderInt(p, "Steps / frame", self.iters, lambda v: setattr(self, "iters", v), 1, 50)
        spy.ui.SliderInt(p, "Batch (n x n)", self.batch, lambda v: setattr(self, "batch", v), 16, 128)

    def _set(self, v):
        self.idx = v
        self.reset()

    def _target(self, name):
        self.target_name = name
        self.reset()

    def preset(self, d):
        self._set((self.idx + d) % len(self.PRESETS))
        self.lab.rebuild_ui()

    def reset(self):
        img = self.lab.image(self.target_name, self.res)
        self.ref_np, self.ref = img, self.lab.tensor(img)
        self.net = TextureNet(self.device, self.PRESETS[self.idx][1])
        self.out, self.hist, self.cur, self.t_step = None, [], 0.0, 0.0

    def update(self):
        if self.paused:
            return
        t0 = time.perf_counter()
        self.net.train(self.ref, self.iters, self.batch)
        self.out = self.net.render(self.res, self.res, self.out)
        o = self.out.to_numpy()
        self.t_step = (time.perf_counter() - t0) / self.iters * 1000
        self.cur = psnr(o, self.ref_np)
        self.hist.append((self.net.steps, self.cur))

    def panes(self):
        s, y, g = 364, 96, 30
        x = [VIEW_X + i * (s + g) for i in range(3)]
        L = self.lab
        img, (lo, hi) = chart([(self.hist, COLORS[0])], 3 * s + 2 * g, 250)
        return [
            ("Reference", f"{self.target_name}  {self.res}x{self.res}", x[0], y, s, s, lambda r: L.show(self.ref, r)),
            ("Network output", f"PSNR {self.cur:5.2f} dB   step {self.net.steps}", x[1], y, s, s,
             lambda r: L.show(_v(self.out, self.ref), r)),
            ("Error x4", "|output - reference|", x[2], y, s, s,
             lambda r: L.show(_v(self.out, self.ref), r, mode=1, ref=self.ref, gain=4)),
            ("PSNR over training", f"{lo:.0f} to {hi:.0f} dB", x[0], y + s + CAPTION_H + 22, 3 * s + 2 * g, 250,
             lambda r: L.show(L.tensor(img, "fitchart"), r)),
        ]

    def stats(self):
        n, c = self.net, self.net.cfg
        return (f"{self.PRESETS[self.idx][0]}\n\nNetwork   {c.describe()}\nWeights   {n.n_weights:,}\n"
                f"Latents   {n.n_latents:,}\nStep      {n.steps:,}\nPSNR      {self.cur:.2f} dB\n"
                f"Time/step {self.t_step:.2f} ms (batch {self.batch * self.batch:,})\n\n"
                + wrap(self.PRESETS[self.idx][2]))


class CompareDemo(Demo):
    title = "Four networks, one budget"
    part = "Part 1 · The idea"
    deck = "Frequency encoding · Adding latent textures"
    blurb = ("Four networks train side by side on the same pixels with the same optimiser. Only "
             "the input and the activation differ.")
    NETS = [("Raw uv, ReLU", NetConfig(enc=ENC_RAW, act=ACT_RELU, seed=3)),
            ("sin/cos encoding", NetConfig(enc=ENC_FREQ, act=ACT_LEAKY, seed=3)),
            ("Latent grid 32x32x8", NetConfig(enc=ENC_LATENT, act=ACT_LEAKY, seed=3)),
            ("SIREN", NetConfig(enc=ENC_RAW, act=ACT_SINE, lr=0.0006, seed=3))]

    def __init__(self, lab):
        super().__init__(lab)
        self.target_name, self.res, self.iters = "slangstars.png", 128, 6

    def build_ui(self, p):
        spy.ui.ComboBox(p, "Target", list(IMAGES).index(self.target_name),
                        lambda v: (setattr(self, "target_name", list(IMAGES)[v]), self.reset()), list(IMAGES))
        spy.ui.SliderInt(p, "Steps / frame", self.iters, lambda v: setattr(self, "iters", v), 1, 30)

    def reset(self):
        img = self.lab.image(self.target_name, self.res)
        self.ref_np, self.ref = img, self.lab.tensor(img)
        self.nets = [TextureNet(self.device, c) for _, c in self.NETS]
        self.outs = [None] * 4
        self.hist = [[] for _ in self.NETS]
        self.cur = [0.0] * 4

    def update(self):
        if self.paused:
            return
        for i, n in enumerate(self.nets):
            n.train(self.ref, self.iters, 64)
            self.outs[i] = n.render(self.res, self.res, self.outs[i])
            self.cur[i] = psnr(self.outs[i].to_numpy(), self.ref_np)
            self.hist[i].append((n.steps, self.cur[i]))

    def panes(self):
        s, gx, y0 = 330, 81, 96
        x = [VIEW_X + i * (s + gx) for i in range(3)]
        y1 = y0 + s + CAPTION_H + 26
        L = self.lab
        img, (lo, hi) = chart([(h, COLORS[i]) for i, h in enumerate(self.hist)], s, s)
        cols = ["mint", "orange", "blue", "yellow"]
        pos = [(x[1], y0), (x[2], y0), (x[0], y1), (x[1], y1)]
        out = [("Reference", self.target_name, x[0], y0, s, s, lambda r: L.show(self.ref, r))]
        for i, (name, _) in enumerate(self.NETS):
            out.append((f"{name}  ({cols[i]})", f"PSNR {self.cur[i]:5.2f} dB",
                        pos[i][0], pos[i][1], s, s, lambda r, i=i: L.show(_v(self.outs[i], self.ref), r)))
        out.append(("PSNR over training", f"{lo:.0f} to {hi:.0f} dB, step {self.nets[0].steps}",
                    x[2], y1, s, s, lambda r: L.show(L.tensor(img, "cmpchart"), r)))
        return out

    def stats(self):
        return ("Step " + f"{self.nets[0].steps:,}\n\n" + "\n".join(
            f"{name}\n   {net.cfg.describe()}\n   {net.n_weights:,} weights, {net.n_latents:,} latents\n"
            f"   PSNR {p:.2f} dB" for (name, _), net, p in zip(self.NETS, self.nets, self.cur)))


class OptimDemo(Demo):
    title = "SGD vs Adam"
    part = "Part 1 · The idea"
    deck = "Rendering as optimisation"
    blurb = ("The same network and the same random pixels, two optimisers. Each step sees a "
             "small random batch, so the gradient is noisy and parameters differ wildly in scale. "
             "Adam keeps running averages of the gradient and its square and gives every "
             "parameter its own step size.")

    def __init__(self, lab):
        super().__init__(lab)
        self.res, self.iters, self.batch = 128, 10, 24
        self.lr_sgd, self.lr_adam = 0.3, 0.005

    def build_ui(self, p):
        spy.ui.SliderFloat(p, "SGD lr", self.lr_sgd, lambda v: (setattr(self, "lr_sgd", v), self.reset()), 0.01, 2.0, "%.2f")
        spy.ui.SliderFloat(p, "Adam lr", self.lr_adam, lambda v: (setattr(self, "lr_adam", v), self.reset()), 0.0005, 0.02, "%.4f")
        spy.ui.SliderInt(p, "Batch (n x n)", self.batch, lambda v: (setattr(self, "batch", v), self.reset()), 8, 64)

    def reset(self):
        img = self.lab.image("slangstars.png", self.res)
        self.ref_np, self.ref = img, self.lab.tensor(img)
        self.nets = [TextureNet(self.device, NetConfig(enc=ENC_FREQ, optimizer="sgd", lr=self.lr_sgd, seed=5)),
                     TextureNet(self.device, NetConfig(enc=ENC_FREQ, optimizer="adam", lr=self.lr_adam, seed=5))]
        self.outs, self.hist, self.cur = [None, None], [[], []], [0.0, 0.0]

    def update(self):
        if self.paused:
            return
        for i, n in enumerate(self.nets):
            n.train(self.ref, self.iters, self.batch)
            self.outs[i] = n.render(self.res, self.res, self.outs[i])
            self.cur[i] = psnr(self.outs[i].to_numpy(), self.ref_np)
            self.hist[i].append((n.steps, self.cur[i]))

    def panes(self):
        s, g, y = 364, 30, 96
        x = [VIEW_X + i * (s + g) for i in range(3)]
        L = self.lab
        img, (lo, hi) = chart([(self.hist[0], COLORS[1]), (self.hist[1], COLORS[0])], 3 * s + 2 * g, 250)
        return [
            ("Reference", "slangstars.png", x[0], y, s, s, lambda r: L.show(self.ref, r)),
            ("Plain gradient descent (orange)", f"lr {self.lr_sgd:.2f}   PSNR {self.cur[0]:5.2f} dB", x[1], y, s, s,
             lambda r: L.show(_v(self.outs[0], self.ref), r)),
            ("Adam (mint)", f"lr {self.lr_adam:.4f}   PSNR {self.cur[1]:5.2f} dB", x[2], y, s, s,
             lambda r: L.show(_v(self.outs[1], self.ref), r)),
            ("PSNR over training", f"{lo:.0f} to {hi:.0f} dB, step {self.nets[0].steps}", x[0], y + s + CAPTION_H + 22,
             3 * s + 2 * g, 250, lambda r: L.show(L.tensor(img, "optchart"), r)),
        ]

    def stats(self):
        return (f"Network   {self.nets[0].cfg.describe()}\nBatch     {self.batch * self.batch} pixels / step\n"
                f"Step      {self.nets[0].steps:,}\n\nSGD   PSNR {self.cur[0]:.2f} dB\nAdam  PSNR {self.cur[1]:.2f} dB\n\n"
                + wrap("Raise the SGD learning rate and it oscillates or diverges; lower it and it crawls. "
                       "Adam's update is the optimizer_step function in the course's network/*.slang."))


class NtcDemo(Demo):
    title = "Neural compression vs BCn"
    part = "Part 2 · Neural Texture Compression"
    deck = "Baseline: BCn · Quality · Training loop · Three ways to deploy"
    blurb = ("PavingStones from the course repo: albedo and normal map. Baseline: BC1 for albedo "
             "(4 bpp) plus BC5 for the normal map (8 bpp), each texture on its own. Neural: one "
             "quantised latent grid for all six channels and a shared decoder, trained with "
             "quantisation noise as NTC does.")
    GRIDS, CHANS, BITS = [32, 64, 128], [2, 4, 8, 16], [2, 4, 6, 8]

    def __init__(self, lab):
        super().__init__(lab)
        self.res, self.gi, self.ci, self.bi = 256, 2, 2, 0
        self.lit, self.iters, self.angle = False, 10, 0.0

    def build_ui(self, p):
        spy.ui.ComboBox(p, "Latent grid", self.gi, lambda v: (setattr(self, "gi", v), self.reset()), [f"{g} x {g}" for g in self.GRIDS])
        spy.ui.ComboBox(p, "Channels", self.ci, lambda v: (setattr(self, "ci", v), self.reset()), [str(c) for c in self.CHANS])
        spy.ui.ComboBox(p, "Bits", self.bi, lambda v: (setattr(self, "bi", v), self.reset()), [str(b) for b in self.BITS])
        spy.ui.SliderInt(p, "Steps / frame", self.iters, lambda v: setattr(self, "iters", v), 1, 100)
        spy.ui.CheckBox(p, "Lit view (moving light)", self.lit, lambda v: setattr(self, "lit", v))

    @property
    def bpp(self):
        g, c, b = self.GRIDS[self.gi], self.CHANS[self.ci], self.BITS[self.bi]
        return g * g * c * b / (self.res * self.res)

    def reset(self):
        lab = self.lab
        a = lab.image("PavingStones albedo", self.res)
        n = lab.image("PavingStones normal", self.res)
        self.ref_np = np.concatenate([a, n], -1)
        self.ref = lab.tensor(self.ref_np)
        key = ("bcn", self.res)
        if key not in lab.cache:
            lab.cache[key] = np.concatenate([bcn.bc1(a), bcn.bc5_normal(n)], -1).astype(np.float32)
        self.bc_np = lab.cache[key]
        self.bc = lab.tensor(self.bc_np)
        self.bc_p = (psnr(self.bc_np[..., :3], a), psnr(self.bc_np[..., 3:], n), psnr(self.bc_np, self.ref_np))
        cfg = NetConfig(enc=ENC_LATENT, out=6, grid=self.GRIDS[self.gi], channels=self.CHANS[self.ci],
                        qbits=self.BITS[self.bi], lr=0.005, lr_grid=0.05, seed=2)
        self.net = TextureNet(self.device, cfg)
        self.out, self.cur, self.hist = None, (0, 0, 0), []

    def update(self):
        self.angle += 0.02
        if self.paused:
            return
        self.net.train(self.ref, self.iters, 64)
        self.out = self.net.render(self.res, self.res, self.out)
        o = np.clip(self.out.to_numpy(), 0, 1)
        self.cur = (psnr(o[..., :3], self.ref_np[..., :3]), psnr(o[..., 3:], self.ref_np[..., 3:]), psnr(o, self.ref_np))
        self.hist.append((self.net.steps, self.cur[2]))

    def panes(self):
        L, out = self.lab, _v(self.out, self.ref)
        light = spy.float3(np.cos(self.angle) * 0.8, np.sin(self.angle) * 0.8, 0.6)
        if self.lit:
            s, g, y = 364, 30, 96
            x = [VIEW_X + i * (s + g) for i in range(3)]
            img, (lo, hi) = chart([(self.hist, COLORS[0])], 3 * s + 2 * g, 250, hline=(self.bc_p[2], (150, 160, 165)))
            return [
                ("Reference, lit", "albedo + normal, 32 bpp raw", x[0], y, s, s, lambda r: L.show(self.ref, r, mode=2, light=light)),
                ("BC1 + BC5, lit", f"12 bpp   PSNR {self.bc_p[2]:.2f} dB", x[1], y, s, s, lambda r: L.show(self.bc, r, mode=2, light=light)),
                ("Neural, lit", f"{self.bpp:.2f} bpp   PSNR {self.cur[2]:.2f} dB", x[2], y, s, s, lambda r: L.show(out, r, mode=2, light=light)),
                ("PSNR, all six channels", f"dashed = BCn   {lo:.0f} to {hi:.0f} dB   step {self.net.steps}",
                 x[0], y + s + CAPTION_H + 22, 3 * s + 2 * g, 250, lambda r: L.show(L.tensor(img, "ntcchart"), r)),
            ]
        s, g = 268, 22
        x = [VIEW_X + i * (s + g) for i in range(4)]
        y0, y1 = 96, 96 + s + CAPTION_H + 28
        res = []
        for row, (name, ch, i) in enumerate([("Albedo", 0, 0), ("Normal", 3, 1)]):
            y = y0 if row == 0 else y1
            res += [
                (f"{name}: reference", "", x[0], y, s, s, lambda r, ch=ch: L.show(self.ref, r, ch=ch)),
                (f"{name}: {'BC1 4 bpp' if ch == 0 else 'BC5 8 bpp'}", f"PSNR {self.bc_p[i]:.2f} dB", x[1], y, s, s,
                 lambda r, ch=ch: L.show(self.bc, r, ch=ch)),
                (f"{name}: neural", f"PSNR {self.cur[i]:.2f} dB", x[2], y, s, s, lambda r, ch=ch: L.show(out, r, ch=ch)),
                (f"{name}: neural error x4", "", x[3], y, s, s,
                 lambda r, ch=ch: L.show(out, r, mode=1, ch=ch, ref=self.ref, gain=4)),
            ]
        return res

    def stats(self):
        w_bits = self.net.n_weights * 16
        ratio = 12 / self.bpp
        return (f"Material  {self.res}x{self.res}, 6 channels\nStep      {self.net.steps:,}\n\n"
                f"             bpp     PSNR all\nBC1 + BC5   12.00   {self.bc_p[2]:6.2f} dB\n"
                f"Neural     {self.bpp:6.2f}   {self.cur[2]:6.2f} dB\n\n"
                f"Neural: {ratio:.1f}x fewer bits, {self.cur[2] - self.bc_p[2]:+.2f} dB vs BCn\n"
                f"Normal map alone: {self.cur[1] - self.bc_p[1]:+.2f} dB vs BC5\n\n"
                + wrap(f"Not counted: the decoder, {self.net.n_weights:,} FP16 weights. On this {self.res}² crop "
                       f"that is {w_bits / self.res ** 2:.2f} bpp; on one 4k material {w_bits / 4096 ** 2:.4f} bpp. "
                       "Compression is training: NTC encodes offline on a GPU.\n\nToy scale: one latent grid, a "
                       "32-wide decoder, a few thousand steps. NTC adds multi-resolution latents and 100k+ "
                       "steps, which is what gets it past BCn at up to 8x fewer bits. The BCn encoder here is "
                       "a simple one; production encoders gain 1-2 dB."))


class DialDemo(Demo):
    title = "Network size is a dial"
    part = "Part 3-4 · GPU execution, neural materials"
    deck = "One layer = matrix x vector · A quality/cost dial"
    blurb = ("Three widths of the same network train together. Every few seconds the lab times "
             "pure inference on this GPU (one million pixels). Bigger buys quality and costs time, "
             "continuously, which no hand-written shader offers.")
    WIDTHS = [16, 32, 64]  # width 64 takes ~15 min to compile the first time (fully unrolled)

    def __init__(self, lab):
        super().__init__(lab)
        self.res, self.iters = 128, 6

    def build_ui(self, p):
        spy.ui.SliderInt(p, "Steps / frame", self.iters, lambda v: setattr(self, "iters", v), 1, 30)
        spy.ui.Button(p, "Time inference now", lambda: self._time())

    def reset(self):
        img = self.lab.image("slangstars.png", self.res)
        self.ref_np, self.ref = img, self.lab.tensor(img)
        self.nets = [TextureNet(self.device, NetConfig(enc=ENC_FREQ, hidden=w, seed=4)) for w in self.WIDTHS]
        self.outs, self.cur, self.ms, self.hist = [None] * 3, [0.0] * 3, [0.0] * 3, [[], [], []]
        self.last_t = 0.0
        self._time()

    def _time(self):
        self.ms = [n.time_inference(512, 3) for n in self.nets]
        self.last_t = time.perf_counter()

    def update(self):
        if self.paused:
            return
        for i, n in enumerate(self.nets):
            n.train(self.ref, self.iters, 64)
            self.outs[i] = n.render(self.res, self.res, self.outs[i])
            self.cur[i] = psnr(self.outs[i].to_numpy(), self.ref_np)
            self.hist[i].append((n.steps, self.cur[i]))
        if time.perf_counter() - self.last_t > 5.0:
            self._time()

    def panes(self):
        s, g, y = 270, 24, 96
        x = [VIEW_X + i * (s + g) for i in range(4)]
        L = self.lab
        img, (lo, hi) = chart([(h, COLORS[i]) for i, h in enumerate(self.hist)], 4 * s + 3 * g, 300)
        res = [("Reference", "slangstars.png", x[0], y, s, s, lambda r: L.show(self.ref, r))]
        cols = ["mint", "orange", "blue"]
        for i, w in enumerate(self.WIDTHS):
            res.append((f"Width {w} ({cols[i]})", f"{self.cur[i]:5.2f} dB  {self.ms[i]:6.2f} ms/Mpx",
                        x[i + 1], y, s, s, lambda r, i=i: L.show(_v(self.outs[i], self.ref), r)))
        res.append(("PSNR over training", f"{lo:.0f} to {hi:.0f} dB   step {self.nets[0].steps}",
                    x[0], y + s + CAPTION_H + 22, 4 * s + 3 * g, 300, lambda r: L.show(L.tensor(img, "dialchart"), r)))
        return res

    def stats(self):
        rows = "\n".join(f"{w:>5}  {n.n_weights:>7,}  {p:6.2f} dB  {m:7.2f} ms"
                         for w, n, p, m in zip(self.WIDTHS, self.nets, self.cur, self.ms))
        return (f"GPU  {self.device.info.adapter_name}\n\nWidth  weights   PSNR      per Mpx\n{rows}\n\n"
                + wrap("Timing is this lab's plain Slang MLP in FP32, no tensor cores. With cooperative "
                       "vectors (FP16 on tensor cores) the course measured over 3x faster training. To run "
                       "that path: hardware-acceleration/mlp-training-coopvec in the course repo."))


class WarpDemo(Demo):
    title = "Warp divergence"
    part = "Part 3 and Part 5"
    deck = "The catch: divergence · The same problem in 3D"
    blurb = ("Tensor cores need one weight matrix for the whole warp. Each 8x4 tile of pixels is "
             "one warp; if its threads hit different materials, or rays in different bricks of a "
             "volume, it runs one pass per distinct network. Sorting threads by network (SER) "
             "removes that cost.")
    WW, WH = 96, 60

    def __init__(self, lab):
        super().__init__(lab)
        self.kind, self.m, self.brick, self.seed, self.auto = 0, 12, 8, 5, False
        self.tick = 0

    def build_ui(self, p):
        spy.ui.ComboBox(p, "Scene", self.kind, lambda v: (setattr(self, "kind", v), self.reset()), ["Materials", "Volume bricks"])
        spy.ui.SliderInt(p, "Materials", self.m, lambda v: (setattr(self, "m", v), self.reset()), 1, 48)
        spy.ui.SliderInt(p, "Brick (px)", self.brick, lambda v: (setattr(self, "brick", v), self.reset()), 2, 24)
        spy.ui.Button(p, "New layout", lambda: (setattr(self, "seed", self.seed + 1), self.reset()))
        spy.ui.CheckBox(p, "Auto-cycle layouts", self.auto, lambda v: setattr(self, "auto", v))

    def warm(self):
        self.reset()

    def reset(self):
        rng = np.random.default_rng(self.seed)
        yy, xx = np.mgrid[0:self.WH, 0:self.WW]
        if self.kind == 0:
            s = rng.uniform([0, 0], [self.WW, self.WH], (self.m, 2))
            d = (xx[..., None] - s[:, 0]) ** 2 + (yy[..., None] - s[:, 1]) ** 2
            ids = d.argmin(-1)
        else:
            a = rng.uniform(0.35, 0.65)
            u = np.floor((xx * np.cos(a) - yy * np.sin(a) + 40) / self.brick).astype(int)
            v = np.floor((xx * np.sin(a) + yy * np.cos(a) + 40) / self.brick).astype(int)
            ids = (u * 7919 + v * 104729) % 997
        self.ids = ids
        warps = ids.reshape(self.WH // 4, 4, self.WW // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 32)
        self.k = np.array([len(set(w)) for w in warps])
        n = ids.size
        self.passes_raster = int(self.k.sum())
        _, counts = np.unique(ids, return_counts=True)
        self.passes_sorted = int(np.ceil(counts / 32).sum())
        self.util = (n / (32 * self.passes_raster), n / (32 * self.passes_sorted))
        self.q_raster = warps
        q = []
        uniq, cnt = np.unique(ids, return_counts=True)
        for v, c in sorted(zip(uniq, cnt), key=lambda t: -t[1]):
            for s in range(0, c, 32):
                row = np.full(32, -1)
                row[:min(32, c - s)] = v
                q.append(row)
        self.q_sorted = np.array(q)
        self._render()

    def _color(self, ids):
        h = (ids * 137.508) % 360 / 360.0
        l = 0.52 + (ids * 7 % 3) * 0.09
        s = 0.38  # HSL -> RGB, same palette as the web version
        c = (1 - np.abs(2 * l - 1)) * s
        hp = h * 6
        x = c * (1 - np.abs(hp % 2 - 1))
        z = np.zeros_like(c)
        out = np.zeros(ids.shape + (3,), np.float32)
        conds = [(hp < 1), (hp >= 1) & (hp < 2), (hp >= 2) & (hp < 3), (hp >= 3) & (hp < 4), (hp >= 4) & (hp < 5), (hp >= 5)]
        trip = [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)]
        for cond, (r, g, b) in zip(conds, trip):
            out[cond] = np.stack([r, g, b], -1)[cond]
        out += (l - c / 2)[..., None]
        return np.clip(out, 0, 1)

    def _render(self):
        cs = 5
        img = np.repeat(np.repeat(self._color(self.ids), cs, 0), cs, 1)
        warn = np.array([232, 145, 79]) / 255
        k = self.k.reshape(self.WH // 4, self.WW // 8)
        for ty in range(k.shape[0]):
            for tx in range(k.shape[1]):
                if k[ty, tx] > 1:
                    y0, x0 = ty * 4 * cs, tx * 8 * cs
                    blk = img[y0:y0 + 4 * cs, x0:x0 + 8 * cs]
                    blk *= 0.7
                    blk[:1], blk[-1:], blk[:, :1], blk[:, -1:] = warn, warn, warn, warn
        self.screen = img.astype(np.float32)
        self.qimg = [self._queue(self.q_raster), self._queue(self.q_sorted)]

    def _queue(self, q):
        rows = 24
        sel = q[np.linspace(0, len(q) - 1, min(rows, len(q))).astype(int)]
        lw, rh = 10, 14
        img = np.empty((rows * rh + 8, 32 * lw + 60, 3), np.float32)
        img[:] = np.array([23, 35, 45]) / 255
        for r, row in enumerate(sel):
            k = len(set(row[row >= 0].tolist()))
            for j, v in enumerate(row):
                col = self._color(np.array([v]))[0] if v >= 0 else np.array([0.2, 0.24, 0.27])
                img[6 + r * rh:6 + r * rh + rh - 3, 6 + j * lw:6 + j * lw + lw - 1] = col
            if k > 1:  # bar showing passes
                img[6 + r * rh:6 + r * rh + rh - 3, 32 * lw + 14:32 * lw + 14 + min(40, 8 * k)] = np.array([232, 145, 79]) / 255
        return img

    def update(self):
        self.tick += 1
        if self.auto and self.tick % 180 == 0:
            self.seed += 1
            self.kind = (self.seed // 2) % 2
            self.reset()

    def panes(self):
        L = self.lab
        sw, sh = 480, 300
        qw, qh = 316, 286
        x0, x1 = VIEW_X, VIEW_X + sw + 20
        y = 96
        return [
            ("Screen: 96 x 60 threads, 8x4 warps", "outlined: warp needs more than one network", x0, y, sw, sh,
             lambda r: L.show(L.tensor(self.screen, "warpscreen"), r)),
            ("Issue queue, screen order", f"{self.passes_raster} passes   utilisation {self.util[0] * 100:.0f}%",
             x1, y, qw, qh, lambda r: L.show(L.tensor(self.qimg[0], "q0"), r)),
            ("Issue queue, sorted by network", f"{self.passes_sorted} passes   utilisation {self.util[1] * 100:.0f}%",
             x1 + qw + 20, y, qw, qh, lambda r: L.show(L.tensor(self.qimg[1], "q1"), r)),
            ("Tensor-core utilisation", "top: screen order   bottom: sorted by network", x0, y + sh + CAPTION_H + 28,
             VIEW_W, 150, lambda r: L.show(L.tensor(self._bars(), "warpbars"), r)),
        ]

    def _bars(self):
        w, h = VIEW_W, 150
        img = np.empty((h, w, 3), np.float32)
        img[:] = np.array([23, 35, 45]) / 255
        for i, (u, col) in enumerate([(self.util[0], (232, 145, 79)), (self.util[1], (92, 207, 191))]):
            y0 = 22 + i * 62
            img[y0:y0 + 40, 16:w - 16] = np.array([42, 58, 68]) / 255
            img[y0:y0 + 40, 16:16 + int((w - 32) * u)] = np.array(col) / 255
        return img

    def stats(self):
        n = self.ids.size
        return (f"Threads      {n:,}\nIdeal warps  {int(np.ceil(n / 32))}\n\n"
                f"Screen order {self.passes_raster:5d} passes  {self.util[0] * 100:4.0f}%\n"
                f"Sorted       {self.passes_sorted:5d} passes  {self.util[1] * 100:4.0f}%\n\n"
                + wrap("Utilisation = useful lanes / (32 x passes). A model of tensor-core issue, not a "
                       "timing. The course measured ~97% of peak when 64+ threads share a matrix and ~1% "
                       "with one matrix per thread (RTX 4090).\n\nVolume bricks stand in for per-block "
                       "networks in volume compression: the brick grid is rotated against the screen, as "
                       "after projection. Queue rows are sampled across the whole frame; orange bars "
                       "count extra passes."))


class VolumeDemo(Demo):
    title = "Neural volume compression"
    part = "Part 5 · What it means for us"
    deck = "The same problem in 3D · HPC view"
    blurb = ("The texture recipe in 3D: a simulation-like density volume (clumps, filaments, "
             "shock shells) stored as a quantised 3D latent grid plus one small MLP, ray-marched "
             "by evaluating the network at every sample. Baseline: a plain dense grid with the "
             "same number of bytes, trilinearly filtered.")
    GRIDS, CHANS = [16, 24, 32], [4, 8]
    N = 128  # reference resolution

    def __init__(self, lab):
        super().__init__(lab)
        self.gi, self.ci, self.iters = 1, 1, 10
        self.rotate, self.show_err, self.angle = True, False, 0.6
        self.size = 364

    def build_ui(self, p):
        spy.ui.ComboBox(p, "Latent grid", self.gi, lambda v: (setattr(self, "gi", v), self.reset()),
                        [f"{g}^3" for g in self.GRIDS])
        spy.ui.ComboBox(p, "Channels", self.ci, lambda v: (setattr(self, "ci", v), self.reset()),
                        [str(c) for c in self.CHANS])
        spy.ui.SliderInt(p, "Steps / frame", self.iters, lambda v: setattr(self, "iters", v), 1, 60)
        spy.ui.CheckBox(p, "Rotate", self.rotate, lambda v: setattr(self, "rotate", v))
        spy.ui.CheckBox(p, "Show error x4", self.show_err, lambda v: setattr(self, "show_err", v))
        spy.ui.Button(p, "Time render modes now", lambda: self._time())

    def _reference(self):
        key = ("volume", self.N)
        if key not in self.lab.cache:
            path = HERE / ".cache" / f"volume{self.N}.npy"
            if path.exists():
                v = np.load(path)
            else:
                print(f"[lab] generating {self.N}^3 volume (once, cached in .cache)")
                v = make_volume(self.N)
                path.parent.mkdir(exist_ok=True)
                np.save(path, v)
            self.lab.cache[key] = (v, self.lab.tensor(v))
        return self.lab.cache[key]

    def reset(self):
        self.ref_np, self.ref = self._reference()
        self.vm = self.lab.volume_module()
        self.cfg = VolConfig(grid=self.GRIDS[self.gi], channels=self.CHANS[self.ci])
        self.net = VolumeNet(self.device, self.cfg)
        self.bytes_dense = self.N ** 3 * 2  # FP16 voxels
        self.bytes_neural = self.cfg.latent_bytes() + self.net.n_weights * 2  # FP16 weights counted
        self.base_res = int(round(self.bytes_neural ** (1 / 3)))  # 8-bit dense grid, same bytes
        base = np.round(resize3(self.ref_np, self.base_res) * 255) / 255
        self.base = self.lab.tensor(base)
        self.base_p = volume_psnr(resize3(base, self.N), self.ref_np)
        self.cur, self.hist, self.frames = 0.0, [], 0
        self.imgs = [None, None, None]
        self.ms = None
        self.decoded = None

    def _cam(self):
        return camera(self.angle)

    def _time(self):
        cam, s, dev = self._cam(), 512, self.device
        tmp = [None]
        def dense():
            tmp[0] = render_dense(self.vm, self.ref, cam, s, tmp[0])
        def neural():
            tmp[0] = self.net.render(cam, s, tmp[0])
        dec = [None]
        def load():
            dec[0] = self.net.decode(self.N, dec[0])
        self.ms = (time_ms(dense, 3, dev), time_ms(neural, 3, dev), time_ms(load, 2, dev))

    def update(self):
        if self.rotate:
            self.angle += 0.01
        if not self.paused:
            self.net.train(self.ref, self.iters, 64)
            self.frames += 1
            if self.frames % 10 == 1:
                self.decoded = self.net.decode(self.N, self.decoded)
                self.cur = volume_psnr(self.decoded.to_numpy(), self.ref_np)
                self.hist.append((self.net.steps, self.cur))
        cam, s = self._cam(), self.size
        self.imgs[0] = render_dense(self.vm, self.ref, cam, s, self.imgs[0])
        self.imgs[1] = render_dense(self.vm, self.base, cam, s, self.imgs[1])
        self.imgs[2] = self.net.render(cam, s, self.imgs[2])
        if self.ms is None:
            self._time()

    def panes(self):
        s, g, y = self.size, 30, 96
        x = [VIEW_X + i * (s + g) for i in range(3)]
        L, e = self.lab, self.show_err
        img, (lo, hi) = chart([(self.hist, COLORS[0])], 3 * s + 2 * g, 250, hline=(self.base_p, (150, 160, 165)))
        kb = lambda b: f"{b / 1024:.0f} KB" if b < 1024 ** 2 else f"{b / 1024 ** 2:.1f} MB"
        n, gr = self.N, self.cfg.grid
        def pane(i):
            if e and i > 0:
                return lambda r: L.show(self.imgs[i], r, mode=1, ref=self.imgs[0], gain=4)
            return lambda r: L.show(self.imgs[i], r)
        return [
            (f"Original {n}^3 FP16", f"{kb(self.bytes_dense)}", x[0], y, s, s, pane(0)),
            (f"Dense {self.base_res}^3 8-bit, same bytes" + (" (error x4)" if e else ""),
             f"{kb(self.bytes_neural)}   PSNR {self.base_p:.2f} dB", x[1], y, s, s, pane(1)),
            (f"Neural {gr}^3x{self.cfg.channels} + MLP" + (" (error x4)" if e else ""),
             f"{kb(self.bytes_neural)}   PSNR {self.cur:.2f} dB", x[2], y, s, s, pane(2)),
            ("Volume PSNR over training", f"dashed = same-bytes dense grid   {lo:.0f} to {hi:.0f} dB   step {self.net.steps}",
             x[0], y + s + CAPTION_H + 22, 3 * s + 2 * g, 250, lambda r: L.show(L.tensor(img, "volchart"), r)),
        ]

    def stats(self):
        ratio = self.bytes_dense / self.bytes_neural
        t = ""
        if self.ms:
            d, ns, ld = self.ms
            t = (f"Render 512^2, this GPU (FP32, no tensor cores)\n"
                 f"  dense grid          {d:7.2f} ms\n"
                 f"  neural on sample    {ns:7.2f} ms   {ns / max(d, 1e-6):.0f}x\n"
                 f"  neural on load      {ld:7.2f} ms once,\n"
                 f"    then dense speed, but {self.bytes_dense / 1024 ** 2:.1f} MB in VRAM\n\n")
        return (f"Network   {self.cfg.describe()}\nLatents   {self.net.n_latents:,} x 8 bit\n"
                f"Weights   {self.net.n_weights:,} x FP16 (counted)\nStep      {self.net.steps:,}\n\n"
                f"Compression {ratio:.0f}x vs FP16 original\n"
                f"Neural {self.cur - self.base_p:+.2f} dB vs same-bytes grid\n\n" + t
                + wrap("Try the grid size: at high ratios (16^3, 24^3) the network wins, at a mild ratio "
                       "(32^3, ~16x) the plain grid is still better. Toy scale: one latent grid, uniform "
                       "voxel sampling, a few thousand steps."))

    def precompile_variants(self):
        for c in self.CHANS:
            VolumeNet._module(self.device, VolConfig(channels=c))


DEMOS = [FitDemo, CompareDemo, OptimDemo, NtcDemo, DialDemo, WarpDemo, VolumeDemo]


# --------------------------------------------------------------------------------------------
# Application shell
# --------------------------------------------------------------------------------------------
class Lab:
    def __init__(self, headless: bool = False, demo: int = 0):
        self.headless = headless
        # Vulkan on purpose: on D3D12 the float atomics used for gradient accumulation go through
        # NVAPI, which DXC rejects for tensors passed inside structs. The course README also
        # lists Vulkan as the Windows backend.
        self.device = make_device(spy.DeviceType.vulkan)
        print(f"[lab] GPU: {self.device.info.adapter_name} ({self.device.info.api_name})")
        self.view = spy.Module.load_from_file(self.device, str(HERE / "lab_view.slang"))
        self.output = self.device.create_texture(
            format=spy.Format.rgba8_unorm, width=WIDTH, height=HEIGHT, mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access | spy.TextureUsage.render_target,
            label="lab_output")
        self.cache: dict = {}
        self._tensors: dict = {}
        if not headless:
            self.window = spy.Window(width=WIDTH, height=HEIGHT, title="Neural Shading Lab", resizable=False)
            self.surface = self.device.create_surface(self.window)
            # Output holds sRGB-encoded values: present through a non-sRGB swapchain format.
            fmts = list(self.surface.info.formats)
            fmt = next((f for f in (spy.Format.bgra8_unorm, spy.Format.rgba8_unorm) if f in fmts), None)
            if fmt is None:
                self.surface.configure(width=WIDTH, height=HEIGHT)
            else:
                self.surface.configure(width=WIDTH, height=HEIGHT, format=fmt)
            self.window.on_keyboard_event = self._on_key
            self.window.on_mouse_event = self._on_mouse
        self.ui = spy.ui.Context(self.device)
        self.panel = spy.ui.Window(self.ui.screen, "Neural Shading Lab", spy.float2(12, 12), spy.float2(PANEL_W, HEIGHT - 24))
        self.demos = [D(self) for D in DEMOS]
        self.captions: list = []
        self.active = -1
        self._pending = None
        self.switch(demo)

    # ---- resources
    def image(self, name, size):
        key = ("img", name, size)
        if key not in self.cache:
            self.cache[key] = IMAGES[name](size)
        return self.cache[key]

    def tensor(self, arr: np.ndarray, slot: str | None = None) -> spy.Tensor:
        """Upload an (H, W, C) float array. With `slot`, reuse one GPU tensor per slot."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if slot is None:
            return spy.Tensor.from_numpy(self.device, arr)
        t = self._tensors.get(slot)
        if t is None or tuple(t.shape) != arr.shape:
            t = spy.Tensor.from_numpy(self.device, arr)
            self._tensors[slot] = t
        else:
            t.copy_from_numpy(arr)
        return t

    def volume_module(self):
        if "volume_module" not in self.cache:
            self.cache["volume_module"] = spy.Module.load_from_file(self.device, str(HERE / "lab_volume.slang"))
        return self.cache["volume_module"]

    def show(self, img: spy.Tensor, rect, mode=0, ch=0, ref=None, gain=4.0, light=None):
        x, y, w, h = rect
        self.view.show(pixel=spy.grid((h, w)), size=spy.int2(w, h), offset=spy.int2(x, y), mode=mode, ch=ch,
                       gain=gain, light=_v(light, spy.float3(0, 0, 1)), img=img, ref=_v(ref, img), output=self.output)

    # ---- UI
    def switch(self, i):
        i = max(0, min(len(self.demos) - 1, i))
        if i != self.active:
            if not getattr(self.demos[i], "_ready", False):
                print(f"[lab] preparing demo {i + 1}: {self.demos[i].title} (first run compiles shaders)")
                self.demos[i].reset()
                self.demos[i]._ready = True
            self.active = i
        self.rebuild_ui()

    @property
    def demo(self) -> Demo:
        return self.demos[self.active]

    def rebuild_ui(self):
        p, d = self.panel, self.demo
        p.remove_all_children()
        # Defer the switch: rebuilding the panel inside a widget callback would edit the widget
        # tree while ImGui is walking it.
        spy.ui.ComboBox(p, "Demo", self.active, lambda v: setattr(self, "_pending", v),
                        [f"{k + 1}  {D.title}" for k, D in enumerate(DEMOS)])
        spy.ui.Text(p, f"\n{d.part}\nDeck: " + wrap(d.deck, 44) + "\n")
        spy.ui.Text(p, wrap(d.blurb) + "\n")
        controls = spy.ui.Group(p, "Controls")
        d.build_ui(controls)
        spy.ui.Button(p, "Pause / resume  (Space)", lambda: setattr(d, "paused", not d.paused))
        spy.ui.Button(p, "Restart  (R)", lambda: d.reset())
        spy.ui.Text(p, "")
        self.stats_text = spy.ui.Text(p, "")
        spy.ui.Text(p, "\nKeys: 1-7 demo, Space pause, R restart,\nLeft/Right preset, F2 screenshot, Esc quit")

    def _caption(self, i, title, sub, x, y, w):
        while len(self.captions) <= i:
            self.captions.append((spy.ui.Window(self.ui.screen, f"cap{len(self.captions)}"), None))
        win, txt = self.captions[i]
        if txt is None:
            txt = spy.ui.Text(win, "")
            self.captions[i] = (win, txt)
        win.title = f"{title}###cap{i}"
        win.position = spy.float2(x, y)
        win.size = spy.float2(w, CAPTION_H)
        win.visible = True
        txt.text = sub

    # ---- frame
    def frame(self):
        if self._pending is not None:
            i, self._pending = self._pending, None
            self.switch(i)
        d = self.demo
        d.update()
        self.view.clear(pixel=spy.grid((HEIGHT, WIDTH)), color=spy.float3(*BG), output=self.output)
        panes = d.panes()
        for i, (title, sub, x, y, w, h, fn) in enumerate(panes):
            self._caption(i, title, sub, x, y, w)
            fn((x, y + CAPTION_H, w, h))
        for j in range(len(panes), len(self.captions)):
            self.captions[j][0].visible = False
        self.stats_text.text = d.stats()

    def render_ui(self, target):
        enc = self.device.create_command_encoder()
        self.ui.begin_frame(WIDTH, HEIGHT)
        self.ui.end_frame(target, enc)
        self.device.submit_command_buffer(enc.finish())

    def run(self):
        while not self.window.should_close():
            self.window.process_events()
            self.frame()
            self.render_ui(self.output)
            image = self.surface.acquire_next_image()
            if not image:
                continue
            enc = self.device.create_command_encoder()
            enc.blit(image, self.output)
            enc.set_texture_state(image, spy.ResourceState.present)
            self.device.submit_command_buffer(enc.finish())
            del image
            self.surface.present()

    def screenshot(self, path: str):
        bmp = self.output.to_bitmap()
        bmp.convert(spy.Bitmap.PixelFormat.rgb, spy.Bitmap.ComponentType.uint8, srgb_gamma=False).write(path)
        print(f"[lab] saved {path}")

    # ---- events
    def _on_key(self, e: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(e):
            return
        if e.type != spy.KeyboardEventType.key_press:
            return
        k = e.key
        if k == spy.KeyCode.escape:
            self.window.close()
        elif k == spy.KeyCode.space:
            self.demo.paused = not self.demo.paused
        elif k == spy.KeyCode.r:
            self.demo.reset()
        elif k == spy.KeyCode.left:
            self.demo.preset(-1)
        elif k == spy.KeyCode.right:
            self.demo.preset(1)
        elif k == spy.KeyCode.f2:
            self.screenshot("screenshot.png")
        else:
            for i in range(len(self.demos)):
                if k == getattr(spy.KeyCode, f"key{i + 1}"):
                    self.switch(i)

    def _on_mouse(self, e: spy.MouseEvent):
        self.ui.handle_mouse_event(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", type=int, default=1, help="demo to open first (1-7)")
    ap.add_argument("--precompile", action="store_true", help="compile all demos' shaders, then open")
    ap.add_argument("--headless", type=int, default=0, metavar="FRAMES",
                    help="no window: run FRAMES frames of --demo and save lab_demo<N>.png (for testing)")
    a = ap.parse_args()
    lab = Lab(headless=bool(a.headless), demo=a.demo - 1)
    if a.precompile:
        for i in range(len(lab.demos)):
            lab.switch(i)
            lab.frame()
            if hasattr(lab.demo, "precompile_variants"):
                lab.demo.precompile_variants()
        lab.switch(a.demo - 1)
    if a.headless:
        for _ in range(a.headless):
            lab.frame()
        lab.render_ui(lab.output)
        lab.screenshot(str(HERE / f"lab_demo{a.demo}.png"))
        return
    lab.run()


if __name__ == "__main__":
    main()
