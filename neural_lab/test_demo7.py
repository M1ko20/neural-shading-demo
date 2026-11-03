# SPDX-License-Identifier: Apache-2.0
"""Headless check of demo 7 on this GPU: compiles the shaders, trains each latent-grid size,
prints volume PSNR against the same-bytes dense grid, times the three render modes, saves renders.

    .venv\\Scripts\\python.exe neural_lab\\test_demo7.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from lab_nn import make_device  # noqa: E402
from lab_volume import (VolConfig, VolumeNet, camera, make_volume, render_dense, resize3,  # noqa: E402
                        time_ms, volume_psnr)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
N = 128

dev = make_device(spy.DeviceType.vulkan)
print(f"GPU: {dev.info.adapter_name} ({dev.info.api_name})", flush=True)
path = HERE / ".cache" / f"volume{N}.npy"
v = np.load(path) if path.exists() else make_volume(N)
vt = spy.Tensor.from_numpy(dev, v)
t0 = time.perf_counter()
vm = spy.Module.load_from_file(dev, str(HERE / "lab_volume.slang"))
print(f"lab_volume.slang compiled in {time.perf_counter() - t0:.1f} s", flush=True)
cam = camera(0.6)
rows, renders = [], [render_dense(vm, vt, cam, 384).to_numpy()]

for g in (16, 24, 32):
    cfg = VolConfig(grid=g, channels=8)
    t0 = time.perf_counter()
    net = VolumeNet(dev, cfg)
    net.train(vt, 1, 64)
    dev.wait()
    t_compile = time.perf_counter() - t0
    nb = cfg.latent_bytes() + net.n_weights * 2
    rb = int(round(nb ** (1 / 3)))
    base = (np.round(resize3(v, rb) * 255) / 255).astype(np.float32)
    bp = volume_psnr(resize3(base, N), v)
    t0 = time.perf_counter()
    for k in range(STEPS // 500):
        net.train(vt, 500, 64)
        dev.wait()
        p = volume_psnr(net.decode(N).to_numpy(), v)
        print(f"  grid {g}^3  step {net.steps:5d}  neural {p:6.2f} dB   same-bytes grid {bp:6.2f} dB", flush=True)
    t_train = (time.perf_counter() - t0) / max(net.steps, 1) * 1000
    out = [None]
    ms_dense = time_ms(lambda: out.__setitem__(0, render_dense(vm, vt, cam, 512, out[0])), 5, dev)
    ms_neural = time_ms(lambda: out.__setitem__(0, net.render(cam, 512, out[0])), 5, dev)
    dec = [None]
    ms_load = time_ms(lambda: dec.__setitem__(0, net.decode(N, dec[0])), 3, dev)
    rows.append((g, N ** 3 * 2 / nb, nb, bp, p, t_compile, t_train, ms_dense, ms_neural, ms_load))
    renders += [render_dense(vm, spy.Tensor.from_numpy(dev, base), cam, 384).to_numpy(), net.render(cam, 384).to_numpy()]

print("\nRESULTS (128^3 FP16 original = 4.0 MB)")
print(" grid  ratio   bytes    grid dB  neural dB   diff  compile s  train ms/step  dense ms  on-sample ms  on-load ms")
for g, r, nb, bp, p, tc, tt, md, mn, ml in rows:
    print(f" {g:3d}^3 {r:5.0f}x {nb / 1024:6.0f}KB  {bp:7.2f}  {p:8.2f}  {p - bp:+6.2f}  {tc:8.1f}  {tt:12.3f}  {md:8.2f}  {mn:11.2f}  {ml:9.2f}")

ref = renders[0]
strip = [np.concatenate([ref, renders[1 + 2 * i], renders[2 + 2 * i]], 1) for i in range(len(rows))]
Image.fromarray((np.clip(np.concatenate(strip, 0), 0, 1) * 255).astype(np.uint8)).save(HERE / "test_demo7.png")
print(f"\nsaved {HERE / 'test_demo7.png'}  (rows: grid 16/24/32; columns: original, same-bytes grid, neural)")
print("DONE")
