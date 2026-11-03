# SPDX-License-Identifier: Apache-2.0
"""Reference BC1 / BC4 / BC5 encoders (vectorised NumPy), decoded back to float images.

Simple but honest encoders: principal-axis endpoints for BC1, min/max for BC4. Production
encoders (e.g. NVTT, bc7enc) search endpoints more carefully and gain roughly 1-2 dB.
"""
import numpy as np


def _blocks(img: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    return img.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)


def _unblocks(b: np.ndarray, h: int, w: int) -> np.ndarray:
    c = b.shape[-1]
    return b.reshape(h // 4, w // 4, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def bc1(rgb: np.ndarray) -> np.ndarray:
    """4 bits per pixel: two RGB565 endpoints and a 2-bit index per pixel, per 4x4 block."""
    h, w, _ = rgb.shape
    px = _blocks(rgb.astype(np.float32))
    mean = px.mean(1, keepdims=True)
    d = px - mean
    cov = np.einsum("bni,bnj->bij", d, d)
    axis = np.ones((px.shape[0], 3), np.float32)
    for _ in range(8):
        axis = np.einsum("bij,bj->bi", cov, axis)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
    t = np.einsum("bni,bi->bn", d, axis)
    e0 = mean[:, 0] + axis * t.min(1, keepdims=True)
    e1 = mean[:, 0] + axis * t.max(1, keepdims=True)
    q = np.array([31, 63, 31], np.float32)
    e0 = np.round(np.clip(e0, 0, 1) * q) / q
    e1 = np.round(np.clip(e1, 0, 1) * q) / q
    pal = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 1)  # (b,4,3)
    dist = ((px[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    idx = dist.argmin(-1)
    out = np.take_along_axis(pal, idx[..., None].repeat(3, -1), 1)
    return _unblocks(out, h, w)


def bc4(ch: np.ndarray) -> np.ndarray:
    """4 bits per pixel, one channel: two 8-bit endpoints and a 3-bit index per pixel."""
    h, w = ch.shape
    px = _blocks(ch[..., None].astype(np.float32))[..., 0]
    lo = np.round(px.min(1, keepdims=True) * 255) / 255
    hi = np.round(px.max(1, keepdims=True) * 255) / 255
    span = np.maximum(hi - lo, 1e-6)
    k = np.round((px - lo) / span * 7)
    out = np.where(hi > lo, lo + span * k / 7, lo)
    return _unblocks(out[..., None], h, w)[..., 0]


def bc5_normal(nrm: np.ndarray) -> np.ndarray:
    """8 bits per pixel: BC4 on X and Y, Z rebuilt in the shader as sqrt(1 - x^2 - y^2)."""
    x = bc4(nrm[..., 0]) * 2 - 1
    y = bc4(nrm[..., 1]) * 2 - 1
    z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
    return np.stack([(x + 1) / 2, (y + 1) / 2, (z + 1) / 2], -1).astype(np.float32)
