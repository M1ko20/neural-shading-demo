# Neural Shading Lab

Live GPU demos for the talk **Neural Shading & Neural Compression**. Everything trains in real time on
your GPU with Slang autodiff and SlangPy, using the same structure as the course samples in
`network/step_0x` (differentiable layers, atomic gradient accumulation, Adam).

## Run

From the repository root (see the top-level README for setting up `.venv`):

```bat
.venv\Scripts\activate
python neural_lab\lab.py --precompile
```

Or just double-click `run_lab.bat` in the repository root.

`--precompile` compiles every demo's shaders once and stores them in `neural_lab\.cache`. The first
run takes a while (about 20 minutes on an RTX 2080, mostly demo 5's width-64 network, plus a few
minutes for demo 7). **Do this before the talk.** Later starts load from the cache and open instantly:

```bat
python neural_lab\lab.py            REM opens on demo 1
python neural_lab\lab.py --demo 4   REM opens on a given demo
```

| Key | Action |
| --- | --- |
| `1` - `7` | switch demo |
| `Space` | pause / resume training |
| `R` | restart the current demo |
| `Left` / `Right` | previous / next preset (demo 1) |
| `F2` | save `screenshot.png` |
| `Esc` | quit |

## Demos and where they go in the deck

| # | Demo | Deck slides | What the audience sees |
| --- | --- | --- | --- |
| 1 | **Fit a texture** | The simplest texture network · Frequency encoding · Adding latent textures | Reference, network output and error map, PSNR curve. Step through presets 1-5 (= course steps 01-05) with the arrow keys, plus SIREN. |
| 2 | **Four networks, one budget** | Frequency encoding · Adding latent textures | Raw uv, sin/cos encoding, latent grid and SIREN training side by side on the same pixels, one PSNR chart. |
| 3 | **SGD vs Adam** | Rendering as optimisation | Same network, same noisy mini-batches, two optimisers. Change the SGD learning rate live to show it crawl or blow up. |
| 4 | **Neural compression vs BCn** | Baseline: BCn · Quality · Training loop | PavingStones albedo + normal map: BC1 + BC5 (12 bpp) against one quantised latent grid (4 bpp by default). Tick *Lit view* for a moving light over all three. |
| 5 | **Network size is a dial** | One layer = matrix × vector · A quality/cost dial | Widths 16 / 32 / 64 train together; inference time per megapixel is measured on this GPU every 5 s. |
| 6 | **Warp divergence** | The catch: divergence · The same problem in 3D | Warps (8×4 tiles) that hit more than one network, the issue queue in screen order vs sorted (SER), utilisation. Switch the scene to *Volume bricks* for the link to our compression work. |
| 7 | **Neural volume compression** | The same problem in 3D · The HPC view | A 128³ simulation-like density volume (clumps, filaments, shock shells) as a quantised 3D latent grid + 32-wide MLP, ray-marched by evaluating the network per sample. Side by side with the original and a dense grid of the same byte size. Times dense vs *on sample* vs *on load*. |

Suggested flow: 1 → 2 after the frequency/latent slides, 3 with *Rendering as optimisation*, 4 at the
end of Part 2, 5 and 6 in Part 3, back to 6 (Volume bricks) and then 7 at *The same problem in 3D*.

## Talking points

**1. Fit a texture.** Start on preset 1 (9 parameters: a gradient), press Right through to 4. On
preset 2 point at the straight edges: ReLU is piecewise linear. Preset 4 is the same network as 3
with a different input: spectral bias. Preset 5 is sharp but read the *Latents* count in the panel:
the detail now lives in the grid, which is exactly what NTC quantises and stores.

**3. SGD vs Adam.** Drag the SGD learning rate up until it oscillates, down until it crawls. Adam
needs no tuning here because every parameter gets its own step size from its gradient history.

**4. Compression.** Turn *Steps / frame* up and let it train for a minute. The panel compares
all six channels at once: at a third of the bits (4 vs 12 bpp) the neural version lands within
about 1 dB of BC1 + BC5 and beats BC5 on the normal map, because albedo and normal share one latent
grid. Be upfront that this is toy scale: one grid, a 32-wide decoder, a few thousand steps, decoder
weights not counted (negligible for a 4k material; the panel prints it). Production NTC adds
multi-resolution latents and 100k+ steps, which is what takes it past BCn at up to 8x fewer bits.
Then tick *Lit view*: under a moving light the differences are hard to see.

**5. Dial.** The timings are a plain FP32 Slang MLP without tensor cores. The cooperative-vector
path is in the course repo: `hardware-acceleration\mlp-training-coopvec` (NVIDIA GPU, Vulkan).

**6. Divergence.** Screen order loses every warp that straddles a boundary; sorting leaves only the
partly filled last warp of each group. *Volume bricks* with small bricks is the per-block-network
case from volume compression: neighbouring rays hit different networks.

**7. Neural volume.** Same recipe as demo 4, one dimension up, on data that looks like ours. Let it
train with *Steps / frame* turned up. Default 24³×8 latents is ~37x smaller than the FP16 original;
the baseline is an honest one: a plain 8-bit dense grid with exactly the same bytes (decoder weights
counted). Then switch *Latent grid*: at 16³ (~117x) and 24³ the network beats the same-bytes grid
(measured ~+1 dB after 3k steps), at 32³ (~16x) the plain grid wins (~35 vs 33 dB). That crossover is
the honest message: neural compression pays off at high ratios. Tick *Show error x4* to see where:
filaments and the sharp shell edges. Press *Time render modes now* for the three deployment modes on
this GPU: dense ray-march, neural *on sample* (small memory, many MLP evaluations per pixel), and
*on load* (decode once, then dense speed but full-size memory). Timings are FP32 without tensor
cores, so *on sample* is the worst case; this is exactly what cooperative vectors and per-brick
sorting (demo 6) are for. The volume is generated once and cached in `.cache\volume128.npy`.

## Files

| File | Contents |
| --- | --- |
| `lab.py` | Window, UI panel, the six demos |
| `lab_nn.py` | `TextureNet`: configurable MLP (raw / frequency / latent input; ReLU / LeakyReLU / sine), training loop, Adam / SGD, inference timing |
| `lab_core.slang` | Differentiable layers, latent grid with quantisation noise, encodings, optimisers |
| `lab_view.slang` | Display kernels: colour, error heat map, lit preview |
| `bcn.py` | Vectorised BC1 / BC4 / BC5 reference encoders |
| `lab_volume.py` | Demo 7: `VolumeNet` (3D latent grid + MLP), procedural volume, same-bytes baseline |
| `lab_volume.slang` | 3D latent grid, volume network, ray marcher for dense and neural fields |

Images come from the SIGGRAPH 2025 course repo (`network/slangstars.png`, `mipmap/PavingStones070_2K.*`, Apache-2.0, The Khronos Group).
Tested with slangpy 0.43.1.
