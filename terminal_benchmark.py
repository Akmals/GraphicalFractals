#!/usr/bin/env python3
"""
terminal_benchmark.py — Sequential vs Parallel Fractal Benchmark (Terminal)

Runs directly in your terminal — no GUI needed.
Measures genuine single-core vs multi-core performance for both fractals.

Usage:
  python3 terminal_benchmark.py                  # full benchmark
  python3 terminal_benchmark.py --live seq        # live sequential FPS counter
  python3 terminal_benchmark.py --live par        # live parallel FPS counter
  python3 terminal_benchmark.py --fractal julia   # benchmark a specific fractal
  python3 terminal_benchmark.py --frames 10       # number of frames for FPS test
  python3 terminal_benchmark.py --res high        # 1200x800 resolution

Controls (during --live mode):
  Ctrl+C  — Stop and show summary
"""

import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

# ─── Configuration ────────────────────────────────────────────────────────────

RESOLUTIONS = {
    "low":    (480,  320),
    "medium": (960,  640),
    "high":   (1200, 800),
}

DEFAULT_BOUNDS = {
    "mandelbrot":   (-2.5,  1.0,  -1.25, 1.25),
    "julia":        (-2.0,  2.0,  -1.5,  1.5),
}

MAX_ITER = 128


# ─── ANSI Styling ─────────────────────────────────────────────────────────────

class Style:
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    CYAN       = "\033[36m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    RED        = "\033[31m"
    MAGENTA    = "\033[35m"
    WHITE      = "\033[97m"
    RESET      = "\033[0m"
    CLEAR_LINE = "\033[2K\r"


def styled(text, *styles):
    return "".join(styles) + str(text) + Style.RESET


# ─── Sequential Renderers (single-core NumPy) ────────────────────────────────

def _mandelbrot_sequential(width, height, bounds, max_iter):
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y = np.linspace(ymin, ymax, height, dtype=np.float64)
    cx, cy = np.meshgrid(x, y)
    zx = np.zeros_like(cx)
    zy = np.zeros_like(cy)
    alive = np.ones((height, width), dtype=bool)
    iters = np.zeros((height, width), dtype=np.float64)

    for _ in range(max_iter):
        ax = zx[alive]; ay = zy[alive]
        zx[alive] = ax * ax - ay * ay + cx[alive]
        zy[alive] = 2.0 * ax * ay + cy[alive]
        iters[alive] += 1.0
        alive &= (zx * zx + zy * zy <= 4.0)
        if not alive.any():
            break

    return _smooth_result(zx, zy, iters, height, width, max_iter)


def _julia_sequential(width, height, bounds, max_iter, jcx=-0.7, jcy=0.27015):
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y = np.linspace(ymin, ymax, height, dtype=np.float64)
    zx, zy = np.meshgrid(x, y)
    zx = zx.copy(); zy = zy.copy()
    alive = np.ones((height, width), dtype=bool)
    iters = np.zeros((height, width), dtype=np.float64)

    for _ in range(max_iter):
        ax = zx[alive]; ay = zy[alive]
        zx[alive] = ax * ax - ay * ay + jcx
        zy[alive] = 2.0 * ax * ay + jcy
        iters[alive] += 1.0
        alive &= (zx * zx + zy * zy <= 4.0)
        if not alive.any():
            break

    return _smooth_result(zx, zy, iters, height, width, max_iter)


def _smooth_result(zx, zy, iters, height, width, max_iter):
    """Apply smooth colouring to escape-time data."""
    result = np.zeros((height, width), dtype=np.float64)
    esc = iters < max_iter
    if esc.any():
        zx2 = zx[esc] ** 2; zy2 = zy[esc] ** 2
        log_zn = np.log(np.maximum(zx2 + zy2, 1e-10)) / 2.0
        nu = np.log(np.maximum(log_zn / np.log(2), 1e-10)) / np.log(2)
        result[esc] = iters[esc] + 1.0 - nu
    return result


# ─── Parallel Worker Functions (top-level for pickling) ───────────────────────

def _parallel_mandelbrot_worker(args):
    start_row, end_row, width, xmin, xmax, ymin, ymax, max_iter, height = args
    chunk_h = end_row - start_row
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y_vals = ymin + np.arange(start_row, end_row) * (ymax - ymin) / height
    cx, cy = np.meshgrid(x, y_vals)
    zx = np.zeros_like(cx); zy = np.zeros_like(cy)
    alive = np.ones((chunk_h, width), dtype=bool)
    iters = np.zeros((chunk_h, width), dtype=np.float64)
    for _ in range(max_iter):
        ax = zx[alive]; ay = zy[alive]
        zx[alive] = ax * ax - ay * ay + cx[alive]
        zy[alive] = 2.0 * ax * ay + cy[alive]
        iters[alive] += 1.0
        alive &= (zx * zx + zy * zy <= 4.0)
        if not alive.any():
            break
    result = np.zeros((chunk_h, width), dtype=np.float64)
    esc = iters < max_iter
    if esc.any():
        zx2 = zx[esc]**2; zy2 = zy[esc]**2
        log_zn = np.log(np.maximum(zx2 + zy2, 1e-10)) / 2.0
        nu = np.log(np.maximum(log_zn / np.log(2), 1e-10)) / np.log(2)
        result[esc] = iters[esc] + 1.0 - nu
    return start_row, result


def _parallel_julia_worker(args):
    start_row, end_row, width, xmin, xmax, ymin, ymax, max_iter, height, jcx, jcy = args
    chunk_h = end_row - start_row
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y_vals = ymin + np.arange(start_row, end_row) * (ymax - ymin) / height
    zx, zy = np.meshgrid(x, y_vals)
    zx = zx.copy(); zy = zy.copy()
    alive = np.ones((chunk_h, width), dtype=bool)
    iters = np.zeros((chunk_h, width), dtype=np.float64)
    for _ in range(max_iter):
        ax = zx[alive]; ay = zy[alive]
        zx[alive] = ax * ax - ay * ay + jcx
        zy[alive] = 2.0 * ax * ay + jcy
        iters[alive] += 1.0
        alive &= (zx * zx + zy * zy <= 4.0)
        if not alive.any():
            break
    result = np.zeros((chunk_h, width), dtype=np.float64)
    esc = iters < max_iter
    if esc.any():
        zx2 = zx[esc]**2; zy2 = zy[esc]**2
        log_zn = np.log(np.maximum(zx2 + zy2, 1e-10)) / 2.0
        nu = np.log(np.maximum(log_zn / np.log(2), 1e-10)) / np.log(2)
        result[esc] = iters[esc] + 1.0 - nu
    return start_row, result


# ─── Parallel Renderer (multiprocessing Pool) ────────────────────────────────

def _render_parallel(fractal_type, width, height, bounds, max_iter, n_workers=None):
    """Render using multiprocessing Pool — genuinely parallel across CPU cores."""
    if n_workers is None:
        n_workers = cpu_count() or 4
    xmin, xmax, ymin, ymax = bounds

    chunk_size = height // n_workers
    tasks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = start + chunk_size if i < n_workers - 1 else height
        if fractal_type == "julia":
            tasks.append((start, end, width, xmin, xmax, ymin, ymax,
                          max_iter, height, -0.7, 0.27015))
        else:
            tasks.append((start, end, width, xmin, xmax, ymin, ymax,
                          max_iter, height))

    worker_fn = {
        "mandelbrot":   _parallel_mandelbrot_worker,
        "julia":        _parallel_julia_worker,
    }[fractal_type]

    result = np.zeros((height, width), dtype=np.float64)
    with Pool(processes=n_workers) as pool:
        for start_row, chunk in pool.map(worker_fn, tasks):
            result[start_row:start_row + chunk.shape[0]] = chunk

    return result


# ─── Dispatch ─────────────────────────────────────────────────────────────────

SEQUENTIAL_FNS = {
    "mandelbrot":   _mandelbrot_sequential,
    "julia":        lambda w, h, b, mi: _julia_sequential(w, h, b, mi),
}


def render_sequential(fractal_type, width, height, bounds, max_iter):
    return SEQUENTIAL_FNS[fractal_type](width, height, bounds, max_iter)


def render_parallel(fractal_type, width, height, bounds, max_iter):
    return _render_parallel(fractal_type, width, height, bounds, max_iter)


# ─── Benchmark Runner ────────────────────────────────────────────────────────

def run_single_benchmark(fractal_type, width, height, bounds, max_iter,
                         num_frames=5):
    """Run a benchmark for one fractal: time sequential and parallel modes."""
    results = {}

    for mode in ("sequential", "parallel"):
        times = []
        render_fn = render_sequential if mode == "sequential" else render_parallel

        # Warmup (small render to fill caches / spawn pool)
        render_fn(fractal_type, width // 4, height // 4, bounds, max_iter // 4)

        for i in range(num_frames):
            t0 = time.perf_counter()
            render_fn(fractal_type, width, height, bounds, max_iter)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            # Progress bar
            bar_len = 20
            filled = int((i + 1) / num_frames * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            color = Style.CYAN if mode == "sequential" else Style.GREEN
            label = styled(mode.upper(), color)
            sys.stdout.write(
                "{clear}    {label}  [{bar}] {n}/{total}  {ms:.0f}ms".format(
                    clear=Style.CLEAR_LINE, label=label, bar=bar,
                    n=i + 1, total=num_frames, ms=elapsed * 1000))
            sys.stdout.flush()

        avg_t = sum(times) / len(times)
        results[mode] = {
            "avg_time": avg_t,
            "min_time": min(times),
            "max_time": max(times),
            "fps":      1.0 / avg_t if avg_t > 0 else 0,
            "times":    times,
        }
        print()  # newline after progress bar

    return results


# ─── Pretty Table ─────────────────────────────────────────────────────────────

def print_comparison_table(all_results, width, height):
    n_cores = cpu_count() or 4

    print()
    print(styled(
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        Style.CYAN, Style.BOLD))
    print(styled(
        "║              SEQUENTIAL vs PARALLEL  —  FRACTAL BENCHMARK                  ║",
        Style.CYAN, Style.BOLD))
    print(styled(
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        Style.CYAN, Style.BOLD))
    print()
    sys_info = "  {s}  {cores} CPU cores  |  {r}  {w}x{h}  |  {m}  {mi}".format(
        s=styled("System:", Style.DIM), cores=n_cores,
        r=styled("Resolution:", Style.DIM), w=width, h=height,
        m=styled("Max iter:", Style.DIM), mi=MAX_ITER)
    print(sys_info)
    print()

    # Header
    hdr = "  {:<16} | {:>12} | {:>12} | {:>8} | {:>8} | {:>8}".format(
        "Fractal", "Sequential", "Parallel", "Speedup", "Seq FPS", "Par FPS")
    div = "  {0} | {1} | {1} | {2} | {2} | {2}".format(
        "─" * 16, "─" * 12, "─" * 8)
    print(styled(hdr, Style.BOLD))
    print(styled(div, Style.DIM))

    for ftype, res in all_results.items():
        seq = res["sequential"]
        par = res["parallel"]
        speedup = seq["avg_time"] / par["avg_time"] if par["avg_time"] > 0 else 0

        if speedup >= 3.0:
            sp_style = (Style.GREEN, Style.BOLD)
        elif speedup >= 1.5:
            sp_style = (Style.YELLOW, Style.BOLD)
        else:
            sp_style = (Style.RED, Style.BOLD)

        name = ftype.replace("_", " ").title()
        par_fps_str = "{:.1f}".format(par["fps"])

        print("  {name:<16} | {seq_ms:>9.1f} ms | {par_ms:>9.1f} ms | "
              "{speedup} | {seq_fps:>6.1f}  | {par_fps}".format(
                  name=styled(name, Style.WHITE),
                  seq_ms=seq["avg_time"] * 1000,
                  par_ms=par["avg_time"] * 1000,
                  speedup=styled("{:>5.1f}x".format(speedup), *sp_style),
                  seq_fps=seq["fps"],
                  par_fps=styled("{:>6}".format(par_fps_str), Style.GREEN, Style.BOLD),
              ))

    print(styled(div, Style.DIM))
    print()

    # Summary
    speedups = [
        r["sequential"]["avg_time"] / r["parallel"]["avg_time"]
        for r in all_results.values()
        if r["parallel"]["avg_time"] > 0
    ]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0

    best = max(all_results.items(),
               key=lambda kv: (kv[1]["sequential"]["avg_time"] /
                               kv[1]["parallel"]["avg_time"])
               if kv[1]["parallel"]["avg_time"] > 0 else 0)
    best_sp = best[1]["sequential"]["avg_time"] / best[1]["parallel"]["avg_time"]

    print("  {label}    {val}  across all fractals using {c} cores".format(
        label=styled("Average Speedup:", Style.BOLD),
        val=styled("{:.1f}x".format(avg_speedup), Style.GREEN, Style.BOLD),
        c=n_cores))
    print("  {label}       {name} ({sp:.1f}x)".format(
        label=styled("Best Speedup:", Style.BOLD),
        name=best[0].replace("_", " ").title(),
        sp=best_sp))
    print()


# ─── Live FPS Mode ────────────────────────────────────────────────────────────

def run_live_fps(fractal_type, mode, width, height, bounds, max_iter):
    """Continuously render and show live FPS in the terminal."""
    render_fn = render_sequential if mode == "seq" else render_parallel
    cores = cpu_count() or 4
    mode_label = "SEQUENTIAL (1 core)" if mode == "seq" else "PARALLEL ({} cores)".format(cores)

    print()
    print(styled(
        "  >> LIVE {} -- {} at {}x{}".format(
            mode_label, fractal_type.replace("_", " ").title(), width, height),
        Style.CYAN, Style.BOLD))
    print(styled("  Press Ctrl+C to stop\n", Style.DIM))

    # Warmup
    sys.stdout.write(styled("  Warming up...", Style.DIM))
    sys.stdout.flush()
    render_fn(fractal_type, width // 4, height // 4, bounds, max_iter // 4)
    print(styled(" done.\n", Style.DIM))

    frame_count = 0
    total_time = 0.0
    fps_history = []
    wall_start = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            render_fn(fractal_type, width, height, bounds, max_iter)
            elapsed = time.perf_counter() - t0

            frame_count += 1
            total_time += elapsed
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0
            avg_fps = frame_count / total_time if total_time > 0 else 0
            fps_history.append(instant_fps)

            # FPS bar
            max_bar_fps = 30.0
            bar_len = 40
            filled = min(int(instant_fps / max_bar_fps * bar_len), bar_len)
            if instant_fps >= 15:
                bar_color = Style.GREEN
            elif instant_fps >= 5:
                bar_color = Style.YELLOW
            else:
                bar_color = Style.RED

            bar_full = styled("█" * filled, bar_color)
            bar_empty = styled("░" * (bar_len - filled), Style.DIM)

            sys.stdout.write(
                "{clr}  Frame {fc}  |  "
                "FPS: {fps}  |  "
                "Avg: {avg}  |  "
                "[{bar_f}{bar_e}]  {ms:.0f}ms".format(
                    clr=Style.CLEAR_LINE,
                    fc=styled(str(frame_count), Style.BOLD),
                    fps=styled("{:6.1f}".format(instant_fps), bar_color, Style.BOLD),
                    avg=styled("{:6.1f}".format(avg_fps), Style.WHITE),
                    bar_f=bar_full, bar_e=bar_empty,
                    ms=elapsed * 1000))
            sys.stdout.flush()

    except KeyboardInterrupt:
        wall_time = time.perf_counter() - wall_start
        print("\n")
        print(styled("  ── Live Session Summary ──\n", Style.CYAN, Style.BOLD))
        print("  Frames Rendered:  {}".format(
            styled(str(frame_count), Style.WHITE, Style.BOLD)))
        print("  Wall Clock Time:  {:.1f}s".format(wall_time))
        if frame_count > 0 and wall_time > 0:
            print("  Average FPS:      {}".format(
                styled("{:.1f}".format(frame_count / wall_time),
                       Style.GREEN, Style.BOLD)))
        if fps_history:
            print("  Best FPS:         {:.1f}".format(max(fps_history)))
            print("  Worst FPS:        {:.1f}".format(min(fps_history)))
            print("  Avg Render Time:  {:.1f}ms".format(
                total_time / frame_count * 1000))
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Sequential vs Parallel Fractal Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 terminal_benchmark.py                       # all fractals, medium res
  python3 terminal_benchmark.py --live seq             # live sequential FPS
  python3 terminal_benchmark.py --live par             # live parallel FPS
  python3 terminal_benchmark.py --fractal mandelbrot   # benchmark one fractal
  python3 terminal_benchmark.py --res high             # 1200x800 resolution
  python3 terminal_benchmark.py --frames 10            # more frames for accuracy
        """,
    )
    parser.add_argument("--live", choices=["seq", "par"],
                        help="Live FPS mode: 'seq' for sequential, 'par' for parallel")
    parser.add_argument("--fractal",
                        choices=["mandelbrot", "julia"],
                        default=None,
                        help="Benchmark a specific fractal (default: all)")
    parser.add_argument("--res",
                        choices=["low", "medium", "high"], default="medium",
                        help="Resolution preset (default: medium = 960x640)")
    parser.add_argument("--frames", type=int, default=5,
                        help="Frames to average for benchmark (default: 5)")

    args = parser.parse_args()
    width, height = RESOLUTIONS[args.res]
    fractals = ([args.fractal] if args.fractal
                else ["mandelbrot", "julia"])

    # Banner
    print()
    print(styled("  ┌─────────────────────────────────────────────────┐",
                 Style.MAGENTA))
    print(styled("  │   ⚡ FRACTAL PERFORMANCE BENCHMARK              │",
                 Style.MAGENTA, Style.BOLD))
    print(styled("  │   Sequential (1 core) vs Parallel (all cores)   │",
                 Style.MAGENTA))
    print(styled("  └─────────────────────────────────────────────────┘",
                 Style.MAGENTA))
    print()
    print("  {c}  {cores}     {r}  {w}x{h}     {m}  {mi}".format(
        c=styled("CPU Cores:", Style.DIM), cores=cpu_count(),
        r=styled("Resolution:", Style.DIM), w=width, h=height,
        m=styled("Max Iter:", Style.DIM), mi=MAX_ITER))
    print()

    if args.live:
        fractal = fractals[0]
        bounds = DEFAULT_BOUNDS[fractal]
        run_live_fps(fractal, args.live, width, height, bounds, MAX_ITER)
        return

    # Full benchmark
    all_results = {}
    for fractal_type in fractals:
        bounds = DEFAULT_BOUNDS[fractal_type]
        name = fractal_type.replace("_", " ").title()
        pad = "─" * max(1, 40 - len(name))
        print(styled("  ── {} {}".format(name, pad), Style.YELLOW, Style.BOLD))
        results = run_single_benchmark(
            fractal_type, width, height, bounds, MAX_ITER, args.frames)
        all_results[fractal_type] = results
        print()

    print_comparison_table(all_results, width, height)


if __name__ == "__main__":
    main()
