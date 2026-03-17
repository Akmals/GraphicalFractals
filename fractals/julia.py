"""
julia.py — Julia Set fractal renderer.

The Julia set uses a fixed complex parameter `c` (instead of mapping
each pixel to c like Mandelbrot). Animating `c` in a circle produces
mesmerising morphing patterns.

Parallel strategy: same strip-based multiprocessing as Mandelbrot.
"""

import numpy as np
from multiprocessing import Pool
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend import BACKEND, NUM_WORKERS


def _render_strip(args):
    """Render a horizontal strip of the Julia set in a worker process."""
    width, y_start, y_end, height, xmin, xmax, ymin, ymax, max_iter, cx, cy = args

    result = np.zeros((y_end - y_start, width), dtype=np.float64)

    for row_idx, py in enumerate(range(y_start, y_end)):
        for px in range(width):
            # pixel → complex z (starting point)
            zx = xmin + px * (xmax - xmin) / width
            zy = ymin + py * (ymax - ymin) / height

            iteration = 0
            while zx * zx + zy * zy <= 4.0 and iteration < max_iter:
                zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
                iteration += 1

            if iteration < max_iter:
                log_zn = np.log(zx * zx + zy * zy) / 2.0
                nu = np.log(log_zn / np.log(2)) / np.log(2)
                result[row_idx, px] = iteration + 1 - nu
            else:
                result[row_idx, px] = 0.0

    return y_start, result


def _render_sequential(width, height, bounds, max_iter, c):
    xmin, xmax, ymin, ymax = bounds
    args = (width, 0, height, height, xmin, xmax, ymin, ymax, max_iter, c.real, c.imag)
    _, result = _render_strip(args)
    return result


def _render_multiprocessing(width, height, bounds, max_iter, c):
    xmin, xmax, ymin, ymax = bounds
    strip_height = height // NUM_WORKERS

    strips = []
    for i in range(NUM_WORKERS):
        y_start = i * strip_height
        y_end = height if i == NUM_WORKERS - 1 else y_start + strip_height
        strips.append((width, y_start, y_end, height, xmin, xmax, ymin, ymax,
                        max_iter, c.real, c.imag))

    result = np.zeros((height, width), dtype=np.float64)

    with Pool(processes=NUM_WORKERS) as pool:
        for y_start, strip_data in pool.map(_render_strip, strips):
            result[y_start: y_start + strip_data.shape[0]] = strip_data

    return result


def render(width, height, bounds, max_iter=256, c=None, force_sequential=False):
    """
    Render the Julia set for complex parameter c.
    Default c is a visually interesting starting value.
    Animate c by calling render() repeatedly with changing c.
    """
    if c is None:
        c = complex(-0.7, 0.27015)

    if force_sequential or BACKEND == "SEQUENTIAL":
        return _render_sequential(width, height, bounds, max_iter, c)
    elif BACKEND == "MULTIPROCESSING":
        return _render_multiprocessing(width, height, bounds, max_iter, c)
    elif BACKEND == "CUDA":
        raise NotImplementedError("Switch to PC to use CUDA backend.")
    else:
        return _render_sequential(width, height, bounds, max_iter, c)


def get_animated_c(t: float, radius: float = 0.7885) -> complex:
    """
    Returns c on a circle of given radius in the complex plane.
    t is time in seconds. Produces a looping Julia animation.
    """
    angle = t * 0.5  # radians per second
    return complex(radius * math.cos(angle), radius * math.sin(angle))
