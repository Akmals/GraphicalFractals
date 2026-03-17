"""
benchmark.py — CPU Sequential vs CPU Parallel timing comparison.

Can be run standalone from terminal:
  python benchmark.py

Or triggered from inside the app by pressing P.
It renders the same fractal twice and reports the speedup.
"""

import time
import numpy as np
from fractals.mandelbrot import render as mandelbrot_render


def run_benchmark(width=960, height=540,
                  bounds=(-2.5, 1.0, -1.25, 1.25),
                  max_iter=256,
                  fractal_render_fn=None):
    """
    Time sequential vs parallel rendering.

    Args:
        width, height: render resolution
        bounds:        (xmin, xmax, ymin, ymax) complex plane window
        max_iter:      max escape iterations
        fractal_render_fn: the fractal's render() function to use

    Returns:
        dict with seq_time, par_time, speedup, seq_arr, par_arr
    """
    if fractal_render_fn is None:
        fractal_render_fn = mandelbrot_render

    print(f"\n{'─'*52}")
    print(f"  Benchmark: {width}×{height} | max_iter={max_iter}")
    print(f"{'─'*52}")

    # ── Sequential (single-process) ───────────────────────────────────────
    print("  Running SEQUENTIAL render...", end=" ", flush=True)
    t0 = time.perf_counter()
    seq_arr = fractal_render_fn(width, height, bounds, max_iter,
                                force_sequential=True)
    seq_time = time.perf_counter() - t0
    print(f"{seq_time:.3f}s")

    # ── Parallel (multiprocessing) ────────────────────────────────────────
    print("  Running PARALLEL render...", end="  ", flush=True)
    t0 = time.perf_counter()
    par_arr = fractal_render_fn(width, height, bounds, max_iter)
    par_time = time.perf_counter() - t0
    print(f"{par_time:.3f}s")

    speedup = seq_time / par_time if par_time > 0 else float("inf")

    # Verify correctness: max allowed difference (float precision)
    diff = np.max(np.abs(seq_arr - par_arr))

    print(f"\n  Sequential : {seq_time:.3f}s")
    print(f"  Parallel   : {par_time:.3f}s")
    print(f"  Speedup    : {speedup:.2f}x")
    print(f"  Max pixel Δ: {diff:.4f}  (0 = identical)")
    print(f"{'─'*52}\n")

    return {
        "seq_time": seq_time,
        "par_time": par_time,
        "speedup": speedup,
        "seq_arr": seq_arr,
        "par_arr": par_arr,
    }


def format_hud_message(result: dict) -> str:
    """Return a 2-line string suitable for the HUD overlay."""
    return (
        f"Sequential: {result['seq_time']:.2f}s  |  "
        f"Parallel: {result['par_time']:.2f}s\n"
        f"Speedup: {result['speedup']:.1f}x  (multiprocessing Pool)"
    )


if __name__ == "__main__":
    run_benchmark()
