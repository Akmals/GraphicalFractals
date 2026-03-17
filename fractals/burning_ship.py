"""
burning_ship.py — Burning Ship fractal renderer.

Uses absolute values on the imaginary/real parts before squaring:
  z ← (|Re(z)| + i|Im(z)|)² + c

This creates a fractal that resembles a burning ship when viewed upside down
(we flip vertically automatically).

Parallel strategy: same strip-based multiprocessing as Mandelbrot.
"""

import numpy as np
from multiprocessing import Pool
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend import BACKEND, NUM_WORKERS


def _render_strip(args):
    """Render a horizontal strip of the Burning Ship fractal in a worker process."""
    width, y_start, y_end, height, xmin, xmax, ymin, ymax, max_iter = args

    result = np.zeros((y_end - y_start, width), dtype=np.float64)

    for row_idx, py in enumerate(range(y_start, y_end)):
        for px in range(width):
            cx = xmin + px * (xmax - xmin) / width
            cy = ymin + py * (ymax - ymin) / height

            zx, zy = 0.0, 0.0
            iteration = 0
            while zx * zx + zy * zy <= 4.0 and iteration < max_iter:
                # The key difference: take absolute values before squaring
                zx, zy = zx * zx - zy * zy + cx, 2.0 * abs(zx) * abs(zy) + cy
                iteration += 1

            if iteration < max_iter:
                log_zn = np.log(zx * zx + zy * zy) / 2.0
                nu = np.log(log_zn / np.log(2)) / np.log(2)
                result[row_idx, px] = iteration + 1 - nu
            else:
                result[row_idx, px] = 0.0

    return y_start, result


def _render_sequential(width, height, bounds, max_iter):
    xmin, xmax, ymin, ymax = bounds
    args = (width, 0, height, height, xmin, xmax, ymin, ymax, max_iter)
    _, result = _render_strip(args)
    # Flip vertically so the "ship" faces the right way
    return np.flipud(result)


def _render_multiprocessing(width, height, bounds, max_iter):
    xmin, xmax, ymin, ymax = bounds
    strip_height = height // NUM_WORKERS

    strips = []
    for i in range(NUM_WORKERS):
        y_start = i * strip_height
        y_end = height if i == NUM_WORKERS - 1 else y_start + strip_height
        strips.append((width, y_start, y_end, height, xmin, xmax, ymin, ymax, max_iter))

    result = np.zeros((height, width), dtype=np.float64)

    with Pool(processes=NUM_WORKERS) as pool:
        for y_start, strip_data in pool.map(_render_strip, strips):
            result[y_start: y_start + strip_data.shape[0]] = strip_data

    return np.flipud(result)


def render(width, height, bounds, max_iter=256, force_sequential=False, **kwargs):
    """Render the Burning Ship fractal."""
    if force_sequential or BACKEND == "SEQUENTIAL":
        return _render_sequential(width, height, bounds, max_iter)
    elif BACKEND == "MULTIPROCESSING":
        return _render_multiprocessing(width, height, bounds, max_iter)
    elif BACKEND == "CUDA":
        raise NotImplementedError("Switch to PC to use CUDA backend.")
    else:
        return _render_sequential(width, height, bounds, max_iter)
