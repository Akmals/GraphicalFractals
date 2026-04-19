# CUDA GPU Rendering — PC Setup Guide

This project has three interchangeable render backends:

| Backend             | Where             | Hardware                        |
| ------------------- | ----------------- | ------------------------------- |
| `SEQUENTIAL`        | Single-threaded   | Any CPU (baseline)              |
| `MULTIPROCESSING`   | Numba parallel JIT| Mac / any multi-core CPU        |
| `CUDA`              | Numba `@cuda.jit` | PC + NVIDIA GPU (e.g. 4060 Ti)  |

Both fractals (Mandelbrot, Julia) ship with a CUDA kernel that launches one
GPU thread per pixel in a 2-D grid of 16×16 thread blocks. For a 1200×800
frame that's ≈ 960 000 parallel threads. Since the escape-time loop is
embarrassingly parallel, the GPU path typically beats the Numba CPU path by
**50–200×** (the exact ratio depends on the zoom level and `MAX_ITER`).

---

## 1. Prerequisites on the PC

1. **NVIDIA GPU** with a recent driver. Verify with:
   ```powershell
   nvidia-smi
   ```
   You should see the card (e.g. `RTX 4060 Ti`) and a driver version. If the
   command isn't found, install/update the driver from
   <https://www.nvidia.com/Download/index.aspx>.

2. **Python 3.10 – 3.12** (Numba's CUDA backend tracks these):
   ```powershell
   python --version
   ```

3. **CUDA Toolkit 12.x** — pick one of:
   - **Option A (easy, recommended):** install via pip wheels, already listed
     in `requirements.txt`:
     ```powershell
     pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12
     ```
   - **Option B (full toolkit):** download the installer from
     <https://developer.nvidia.com/cuda-downloads>. This also puts `nvcc` on
     the PATH if you later want to write raw `.cu` kernels.

---

## 2. Install the project

```powershell
cd path\to\GraphicalFractals
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12
```

Confirm Numba sees the GPU:

```powershell
python -c "from numba import cuda; print(cuda.detect())"
```

Expect output listing your GPU with `Summary: 1/1 devices are supported`.

---

## 3. Switch the backend to CUDA

Two equivalent ways:

**A. Edit `backend.py`** default:
```python
BACKEND = os.environ.get("FRACTAL_BACKEND", "CUDA").upper()
```

**B. Use the env var** (no code edit needed):
```powershell
$env:FRACTAL_BACKEND = "CUDA"
python main.py
```

On startup the terminal prints the active backend, e.g.
```
Fractal Explorer starting — backend: GPU CUDA (RTX 4060 Ti)
```

---

## 4. Run it

```powershell
python main.py
```

Controls:

| Key                | Action                     |
| ------------------ | -------------------------- |
| WASD / Arrow keys  | Pan                        |
| + / − / Scroll     | Zoom                       |
| M / J              | Mandelbrot / Julia         |
| C                  | Cycle colour palette       |
| Space              | Toggle Julia animation     |
| R                  | Reset view                 |
| **P**              | **Benchmark CPU vs GPU**   |
| S                  | Screenshot                 |
| H                  | Help overlay               |
| ESC                | Quit                       |

When `BACKEND="CUDA"`, pressing **P** benchmarks the GPU kernel against the
Numba CPU JIT path (benchmark's `force_sequential=True` intentionally skips
CUDA so you get a real CPU-vs-GPU comparison). Expect 50×–200× speedups.

---

## 5. How the CUDA kernel works

Each fractal module (`fractals/mandelbrot.py`, `fractals/julia.py`) defines a
kernel decorated with `@cuda.jit`. The core idea:

```python
@cuda.jit(fastmath=True)
def _mandelbrot_cuda_kernel(result, xmin, ymin, dx, dy, max_iter):
    px, py = cuda.grid(2)          # this thread's pixel
    height, width = result.shape
    if px >= width or py >= height:
        return
    cx = xmin + px * dx
    cy = ymin + py * dy
    zx = zy = 0.0
    i = 0
    while zx*zx + zy*zy <= 4.0 and i < max_iter:
        zx, zy = zx*zx - zy*zy + cx, 2.0*zx*zy + cy
        i += 1
    result[py, px] = i + smoothing_term
```

Launch configuration:
```python
threads_per_block = (16, 16)                     # 256 threads/block
blocks_per_grid   = (ceil(W / 16), ceil(H / 16))
kernel[blocks_per_grid, threads_per_block](d_result, ...)
```

Output lives in GPU memory (`cuda.device_array`) and is copied back to host
with `.copy_to_host()` once per frame.

---

## 6. Troubleshooting

- **`CudaSupportError: Error at driver init`**
  The NVIDIA driver is missing or too old. Update it.

- **`LinkerError: libnvvm not found`**
  The CUDA toolkit isn't visible to Numba. Either `pip install
  nvidia-cuda-nvcc-cu12` (Option A) or set `CUDA_HOME` to point to your
  toolkit install (Option B).

- **`cuda.is_available()` returns False but `nvidia-smi` works**
  Usually a mismatch: toolkit 12.x with an older driver, or 32-bit Python.
  Make sure you're on 64-bit Python 3.10–3.12 and the CUDA 12.x runtime is
  installed.

- **Frame rate lower than expected**
  Increase `MAX_ITER` in `main.py` — the GPU has headroom for 512+ iterations
  while still feeling real-time.
