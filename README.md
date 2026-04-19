# GPU Fractal Explorer

An interactive fractal explorer with three interchangeable render backends:
sequential CPU, parallel CPU (Numba JIT), and CUDA GPU kernels.

VIEW LIVE HERE: https://web-zeta-jet-29.vercel.app/

## Features
- 2 Fractals: Mandelbrot Set, Julia Set (with live animation)
- Three backends: sequential, parallel CPU, **CUDA GPU** (see [CUDA_SETUP.md](CUDA_SETUP.md))
- 12 colour palettes
- Built-in benchmark: press P to compare sequential vs parallel/GPU timing
- Screenshot export: press S

## Quick Start (Mac / CPU)

```bash
pip3 install -r requirements.txt
python3 main.py
```

## Quick Start (PC / CUDA GPU)

See **[CUDA_SETUP.md](CUDA_SETUP.md)** for full instructions. Summary:

```powershell
pip install -r requirements.txt
pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12
$env:FRACTAL_BACKEND = "CUDA"
python main.py
```

## Controls

| Key | Action |
|---|---|
| WASD / Arrow keys | Pan |
| + / - / Scroll | Zoom |
| M | Mandelbrot |
| J | Julia Set |
| C | Cycle palette |
| Space | Toggle Julia animation |
| R | Reset view |
| P | Benchmark (sequential vs parallel/GPU) |
| S | Screenshot |
| H | Help overlay |
| ESC | Quit |

## Project Structure

```
gpu-fractals/
├── main.py              # Pygame app — event loop, rendering, controls
├── backend.py           # Backend selector
├── benchmark.py         # Timing comparison runner
├── fractals/
│   ├── mandelbrot.py    # Mandelbrot set (CPU + CUDA)
│   └── julia.py         # Julia set with animation (CPU + CUDA)
└── renderer/
    ├── colormap.py      # 12 colour palette LUTs
    └── hud.py           # On-screen display overlay
```
