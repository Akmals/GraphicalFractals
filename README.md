# GPU Fractal Explorer

An interactive fractal explorer that uses Python multiprocessing to render fractals in parallel across all CPU cores.

VIEW LIVE HERE: https://web-zeta-jet-29.vercel.app/ 
## Features
- 3 Fractals: Mandelbrot Set, Julia Set (with live animation), Burning Ship
- Parallel rendering using Python multiprocessing
- 12 colour palettes
- Built-in benchmark: press P to compare sequential vs parallel timing
- Screenshot export: press S

## Quick Start

```bash
pip3 install -r requirements.txt
python3 main.py
```

## Controls

| Key | Action |
|---|---|
| WASD / Arrow keys | Pan |
| + / - / Scroll | Zoom |
| M | Mandelbrot |
| J | Julia Set |
| B | Burning Ship |
| C | Cycle palette |
| Space | Toggle Julia animation |
| R | Reset view |
| P | Benchmark (sequential vs parallel) |
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
│   ├── mandelbrot.py    # Mandelbrot set
│   ├── julia.py         # Julia set with animation
│   └── burning_ship.py  # Burning Ship fractal
└── renderer/
    ├── colormap.py      # 12 colour palette LUTs
    └── hud.py           # On-screen display overlay
```
