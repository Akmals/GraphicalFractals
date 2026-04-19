"""
mandelbrot.py — Mandelbrot Set fractal renderer.

Three backends, selected via backend.BACKEND:

  SEQUENTIAL / MULTIPROCESSING
      Numba @njit(parallel=True) on the CPU. One thread per row (prange),
      JIT-compiled to native machine code. Falls back to vectorised NumPy
      when Numba is unavailable.

  CUDA
      Numba @cuda.jit kernel. Launches one GPU thread per pixel in a 2-D
      grid (16×16 thread blocks). A 1200×800 frame is ~960k parallel
      threads — the Mandelbrot escape-time loop is embarrassingly parallel
      so a modern GPU (e.g. RTX 4060 Ti) beats the CPU path by 50–200×.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend import BACKEND

# ─── Try Numba CPU JIT ───────────────────────────────────────────────────────
try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False

# ─── Try Numba CUDA ──────────────────────────────────────────────────────────
_CUDA = False
if BACKEND == "CUDA":
    try:
        from numba import cuda
        import math as _math
        if cuda.is_available():
            _CUDA = True
    except Exception:
        _CUDA = False


# ─── Numba CPU JIT path ──────────────────────────────────────────────────────

if _NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _mandelbrot_numba(width, height, xmin, xmax, ymin, ymax, max_iter):
        result = np.zeros((height, width), dtype=np.float64)
        dx = (xmax - xmin) / width
        dy = (ymax - ymin) / height
        log2 = np.log(2.0)

        for py in prange(height):
            cy = ymin + py * dy
            for px in range(width):
                cx = xmin + px * dx
                zx, zy = 0.0, 0.0
                i = 0
                while zx * zx + zy * zy <= 4.0 and i < max_iter:
                    zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
                    i += 1
                if i < max_iter:
                    log_zn = np.log(zx * zx + zy * zy) * 0.5
                    nu = np.log(log_zn / log2) / log2
                    result[py, px] = i + 1.0 - nu

        return result


# ─── CUDA GPU kernel ─────────────────────────────────────────────────────────
#
# Kernel launch strategy:
#   - 2-D grid of 16×16 thread blocks → 256 threads per block.
#   - Each thread computes exactly one pixel of the output image.
#   - cuda.grid(2) returns the pixel coordinates (px, py) this thread owns.
#   - No inter-thread communication needed: Mandelbrot is embarrassingly
#     parallel, so every pixel is independent → perfect GPU workload.
#
# Memory:
#   - The output array lives in GPU device memory (cuda.device_array).
#   - We copy back to host only once per frame (.copy_to_host()).

if _CUDA:
    @cuda.jit(fastmath=True)
    def _mandelbrot_cuda_kernel(result, xmin, ymin, dx, dy, max_iter):
        px, py = cuda.grid(2)
        height, width = result.shape
        if px >= width or py >= height:
            return

        cx = xmin + px * dx
        cy = ymin + py * dy
        zx = 0.0
        zy = 0.0
        i = 0
        while zx * zx + zy * zy <= 4.0 and i < max_iter:
            zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
            i += 1

        if i < max_iter:
            log_zn = _math.log(zx * zx + zy * zy) * 0.5
            nu = _math.log(log_zn / _math.log(2.0)) / _math.log(2.0)
            result[py, px] = i + 1.0 - nu
        else:
            result[py, px] = 0.0

    def _mandelbrot_cuda(width, height, xmin, xmax, ymin, ymax, max_iter):
        dx = (xmax - xmin) / width
        dy = (ymax - ymin) / height

        d_result = cuda.device_array((height, width), dtype=np.float64)

        # 16×16 threads per block is a good default on every modern CUDA GPU.
        threads_per_block = (16, 16)
        blocks_x = (width  + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        _mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](
            d_result, xmin, ymin, dx, dy, max_iter
        )
        return d_result.copy_to_host()


# ─── NumPy fallback ──────────────────────────────────────────────────────────

def _mandelbrot_numpy(width, height, xmin, xmax, ymin, ymax, max_iter):
    """Vectorised NumPy fallback (no Numba)."""
    x  = np.linspace(xmin, xmax, width,  dtype=np.float64)
    y  = np.linspace(ymin, ymax, height, dtype=np.float64)
    cx, cy = np.meshgrid(x, y)

    zx    = np.zeros_like(cx)
    zy    = np.zeros_like(cy)
    alive = np.ones((height, width), dtype=bool)
    iters = np.zeros((height, width), dtype=np.float32)

    for _ in range(max_iter):
        ax  = zx[alive]; ay = zy[alive]
        ax2 = ax * ax;   ay2 = ay * ay
        zx[alive] = ax2 - ay2 + cx[alive]
        zy[alive] = 2.0 * ax * ay + cy[alive]
        iters[alive] += 1.0
        alive &= (zx * zx + zy * zy <= 4.0)
        if not alive.any():
            break

    result = np.zeros((height, width), dtype=np.float64)
    esc = iters < max_iter
    if esc.any():
        zx2_e  = zx[esc] ** 2; zy2_e = zy[esc] ** 2
        log_zn = np.log(np.maximum(zx2_e + zy2_e, 1e-10)) / 2.0
        nu     = np.log(np.maximum(log_zn / np.log(2), 1e-10)) / np.log(2)
        result[esc] = iters[esc] + 1.0 - nu
    return result


# ─── Public API ──────────────────────────────────────────────────────────────

_warmed_up = False

def render(width, height, bounds, max_iter=256, force_sequential=False):
    """
    Main render entry point. Dispatch order:
      1. CUDA kernel (when BACKEND="CUDA" and a GPU is present)
      2. Numba CPU parallel JIT
      3. Vectorised NumPy
    force_sequential=True is used by the benchmark.
    """
    global _warmed_up
    xmin, xmax, ymin, ymax = bounds

    if _CUDA and not force_sequential:
        if not _warmed_up:
            # Warm-up launch compiles the kernel & allocates context (~1 s once).
            _mandelbrot_cuda(64, 64, xmin, xmax, ymin, ymax, 16)
            _warmed_up = True
        return _mandelbrot_cuda(width, height, xmin, xmax, ymin, ymax, max_iter)

    if _NUMBA:
        if not _warmed_up:
            _mandelbrot_numba(64, 64, xmin, xmax, ymin, ymax, 16)
            _warmed_up = True
        return _mandelbrot_numba(width, height, xmin, xmax, ymin, ymax, max_iter)

    return _mandelbrot_numpy(width, height, xmin, xmax, ymin, ymax, max_iter)
