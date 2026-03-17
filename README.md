# ⚡ GPU Fractal Explorer — Parallel Processing Demo

An interactive fractal explorer that demonstrates **true parallel processing** using Python's `multiprocessing` module (Mac) or CUDA GPU kernels (PC with RTX 4060 Ti).

## Features
- **3 Fractals**: Mandelbrot Set, Julia Set (with live animation), Burning Ship
- **True Parallelism**: Image split into CPU-core-count strips, each rendered in a separate OS process
- **12 Color Palettes**: Inferno, Plasma, Viridis, Magma, and more
- **Built-in Benchmark**: Press `P` to compare sequential vs parallel and see the speedup live
- **Screenshot Export**: Press `S`

## Quick Start

```bash
cd /Users/akmal/.gemini/antigravity/scratch/gpu-fractals
pip3 install -r requirements.txt
python3 main.py
```

## Controls

| Key | Action |
|---|---|
| `WASD` / Arrow keys | Pan |
| `+` / `-` / Scroll | Zoom |
| `M` | Mandelbrot |
| `J` | Julia Set |
| `B` | Burning Ship |
| `C` | Cycle palette |
| `Space` | Toggle Julia animation |
| `R` | Reset view |
| `P` | **Benchmark** (sequential vs parallel) |
| `S` | Screenshot |
| `H` | Help overlay |
| `ESC` | Quit |

## Project Structure

```
gpu-fractals/
├── main.py              # Pygame app — event loop, rendering, controls
├── backend.py           # Backend selector (MULTIPROCESSING or CUDA)
├── benchmark.py         # Timing comparison runner
├── fractals/
│   ├── mandelbrot.py    # Mandelbrot set — sequential + multiprocessing
│   ├── julia.py         # Julia set — sequential + multiprocessing + animation
│   └── burning_ship.py  # Burning Ship — sequential + multiprocessing
├── renderer/
│   ├── colormap.py      # 12 pre-baked colour palette LUTs
│   └── hud.py           # On-screen display overlay
└── screenshots/         # Saved screenshots
```

## Moving to PC (CUDA)

1. Install CUDA Toolkit + `numba` and `cupy-cuda12x`
2. In `backend.py`, change:
   ```python
   BACKEND = "MULTIPROCESSING"
   ```
   to:
   ```python
   BACKEND = "CUDA"
   ```
3. Implement the `render_cuda()` stub in each fractal module using `@cuda.jit`

That's it — the rest of the app works unchanged.
