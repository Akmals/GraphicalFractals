"""
burning_ship.py — Burning Ship fractal renderer.

Uses Numba JIT compilation with parallel=True for native multi-core speed.
Falls back to fast NumPy vectorisation if Numba is unavailable.

Uses absolute values on the imaginary/real parts before squaring:
  z ← (|Re(z)| + i|Im(z)|)² + c

The fractal is flipped vertically so the "ship" faces the right way.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend import BACKEND

try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False


if _NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _burning_ship_numba(width, height, xmin, xmax, ymin, ymax, max_iter):
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
                    # Burning Ship: abs before the cross-term
                    zx, zy = zx * zx - zy * zy + cx, 2.0 * abs(zx) * abs(zy) + cy
                    i += 1
                if i < max_iter:
                    log_zn = np.log(zx * zx + zy * zy) * 0.5
                    nu = np.log(log_zn / log2) / log2
                    result[py, px] = i + 1.0 - nu

        return result


def _burning_ship_numpy(width, height, xmin, xmax, ymin, ymax, max_iter):
    x  = np.linspace(xmin, xmax, width,  dtype=np.float64)
    y  = np.linspace(ymin, ymax, height, dtype=np.float64)
    cx, cy = np.meshgrid(x, y)
    zx = np.zeros_like(cx); zy = np.zeros_like(cy)
    alive = np.ones((height, width), dtype=bool)
    iters = np.zeros((height, width), dtype=np.float32)

    for _ in range(max_iter):
        ax  = zx[alive]; ay = zy[alive]
        zx[alive] = ax * ax - ay * ay + cx[alive]
        zy[alive] = 2.0 * np.abs(ax) * np.abs(ay) + cy[alive]
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

def render(width, height, bounds, max_iter=256, force_sequential=False, **kwargs):
    """Render the Burning Ship fractal."""
    global _warmed_up
    xmin, xmax, ymin, ymax = bounds

    if _NUMBA:
        if not _warmed_up:
            _burning_ship_numba(64, 64, xmin, xmax, ymin, ymax, 16)
            _warmed_up = True
        result = _burning_ship_numba(width, height, xmin, xmax, ymin, ymax, max_iter)
    else:
        result = _burning_ship_numpy(width, height, xmin, xmax, ymin, ymax, max_iter)

    return np.flipud(result)
