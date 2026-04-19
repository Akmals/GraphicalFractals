"""
julia.py — Julia Set fractal renderer.

Three backends, selected via backend.BACKEND:
  CPU:  Numba @njit(parallel=True) over rows, NumPy fallback.
  GPU:  Numba @cuda.jit kernel, one thread per pixel in a 2-D grid.

The Julia set fixes a complex parameter `c` and iterates
  z ← z² + c
with z₀ set to the pixel's complex-plane coordinate. Animating `c`
around a circle produces the familiar morphing-loop animation.
"""

import numpy as np
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend import BACKEND

try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False

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
    def _julia_numba(width, height, xmin, xmax, ymin, ymax, max_iter, cx, cy):
        result = np.zeros((height, width), dtype=np.float64)
        dx = (xmax - xmin) / width
        dy = (ymax - ymin) / height
        log2 = np.log(2.0)

        for py in prange(height):
            zy0 = ymin + py * dy
            for px in range(width):
                zx, zy = xmin + px * dx, zy0
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
# One thread per pixel; 16×16 thread blocks; pixels are fully independent.

if _CUDA:
    @cuda.jit(fastmath=True)
    def _julia_cuda_kernel(result, xmin, ymin, dx, dy, max_iter, cx, cy):
        px, py = cuda.grid(2)
        height, width = result.shape
        if px >= width or py >= height:
            return

        zx = xmin + px * dx
        zy = ymin + py * dy
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

    def _julia_cuda(width, height, xmin, xmax, ymin, ymax, max_iter, cx, cy):
        dx = (xmax - xmin) / width
        dy = (ymax - ymin) / height

        d_result = cuda.device_array((height, width), dtype=np.float64)

        threads_per_block = (16, 16)
        blocks_x = (width  + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_x, blocks_y)

        _julia_cuda_kernel[blocks_per_grid, threads_per_block](
            d_result, xmin, ymin, dx, dy, max_iter, cx, cy
        )
        return d_result.copy_to_host()


# ─── NumPy fallback ──────────────────────────────────────────────────────────

def _julia_numpy(width, height, xmin, xmax, ymin, ymax, max_iter, cx, cy):
    x  = np.linspace(xmin, xmax, width,  dtype=np.float64)
    y  = np.linspace(ymin, ymax, height, dtype=np.float64)
    zx, zy = np.meshgrid(x, y)
    zx = zx.copy(); zy = zy.copy()
    alive = np.ones((height, width), dtype=bool)
    iters = np.zeros((height, width), dtype=np.float32)

    for _ in range(max_iter):
        ax  = zx[alive]; ay = zy[alive]
        zx[alive] = ax * ax - ay * ay + cx
        zy[alive] = 2.0 * ax * ay + cy
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


_warmed_up = False

def render(width, height, bounds, max_iter=256, c=None, force_sequential=False):
    """Render the Julia set for complex parameter c."""
    global _warmed_up
    if c is None:
        c = complex(-0.7, 0.27015)
    xmin, xmax, ymin, ymax = bounds

    if _CUDA and not force_sequential:
        if not _warmed_up:
            _julia_cuda(64, 64, xmin, xmax, ymin, ymax, 16, c.real, c.imag)
            _warmed_up = True
        return _julia_cuda(width, height, xmin, xmax, ymin, ymax, max_iter, c.real, c.imag)

    if _NUMBA:
        if not _warmed_up:
            _julia_numba(64, 64, xmin, xmax, ymin, ymax, 16, c.real, c.imag)
            _warmed_up = True
        return _julia_numba(width, height, xmin, xmax, ymin, ymax, max_iter, c.real, c.imag)

    return _julia_numpy(width, height, xmin, xmax, ymin, ymax, max_iter, c.real, c.imag)


def get_animated_c(t: float, radius: float = 0.7885) -> complex:
    """Returns c on a circle — animate by passing increasing t (seconds)."""
    angle = t * 0.5
    return complex(radius * math.cos(angle), radius * math.sin(angle))
