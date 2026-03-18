"""
julia.py — Julia Set fractal renderer.

Uses Numba JIT compilation with parallel=True for native multi-core speed.
Falls back to fast NumPy vectorisation if Numba is unavailable.

The Julia set uses a fixed complex parameter `c`; each pixel's starting
z is its own position on the complex plane. Animating `c` produces
mesmerising looping animations.
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

    if _NUMBA:
        if not _warmed_up:
            _julia_numba(64, 64, xmin, xmax, ymin, ymax, 16, c.real, c.imag)
            _warmed_up = True
        return _julia_numba(width, height, xmin, xmax, ymin, ymax, max_iter, c.real, c.imag)
    else:
        return _julia_numpy(width, height, xmin, xmax, ymin, ymax, max_iter, c.real, c.imag)


def get_animated_c(t: float, radius: float = 0.7885) -> complex:
    """Returns c on a circle — animate by passing increasing t (seconds)."""
    angle = t * 0.5
    return complex(radius * math.cos(angle), radius * math.sin(angle))
