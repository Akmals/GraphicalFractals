"""
mandelbrot.py — Mandelbrot Set fractal renderer.

Parallel strategy: splits the image into horizontal strips,
renders each strip in a separate OS process (true parallelism, no GIL).

To port to CUDA on PC:
  Replace _render_strip with a @cuda.jit kernel.
"""

import numpy as np
from multiprocessing import Pool
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend import BACKEND, NUM_WORKERS


# ─── Core iteration logic (runs in worker processes) ─────────────────────────

def _render_strip(args):
    """
    Render a horizontal strip of the Mandelbrot set.
    Called in a worker process — safe from the GIL.
    """
    width, y_start, y_end, height, xmin, xmax, ymin, ymax, max_iter = args

    result = np.zeros((y_end - y_start, width), dtype=np.float64)

    for row_idx, py in enumerate(range(y_start, y_end)):
        for px in range(width):
            # Map pixel → complex plane
            cx = xmin + px * (xmax - xmin) / width
            cy = ymin + py * (ymax - ymin) / height

            zx, zy = 0.0, 0.0
            iteration = 0
            while zx * zx + zy * zy <= 4.0 and iteration < max_iter:
                zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
                iteration += 1

            if iteration < max_iter:
                # Smooth coloring — removes banding artifacts
                log_zn = np.log(zx * zx + zy * zy) / 2.0
                nu = np.log(log_zn / np.log(2)) / np.log(2)
                result[row_idx, px] = iteration + 1 - nu
            else:
                result[row_idx, px] = 0.0  # inside set → black

    return y_start, result


def _render_sequential(width, height, bounds, max_iter):
    """Single-process render for benchmarking baseline."""
    xmin, xmax, ymin, ymax = bounds
    args = (width, 0, height, height, xmin, xmax, ymin, ymax, max_iter)
    _, result = _render_strip(args)
    return result


def _render_multiprocessing(width, height, bounds, max_iter):
    """Parallel render using multiprocessing.Pool — true parallelism."""
    xmin, xmax, ymin, ymax = bounds

    # Divide image into strips, one per worker
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

    return result


def render(width, height, bounds, max_iter=256, force_sequential=False):
    """
    Main render entry point. Dispatches based on backend.py setting.
    force_sequential=True is used by the benchmark.
    """
    if force_sequential or BACKEND == "SEQUENTIAL":
        return _render_sequential(width, height, bounds, max_iter)
    elif BACKEND == "MULTIPROCESSING":
        return _render_multiprocessing(width, height, bounds, max_iter)
    elif BACKEND == "CUDA":
        # TODO: import and call numba CUDA kernel here on PC
        raise NotImplementedError("Switch to PC to use CUDA backend.")
    else:
        return _render_sequential(width, height, bounds, max_iter)
