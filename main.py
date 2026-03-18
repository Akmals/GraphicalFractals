"""
main.py — Interactive Fractal Explorer

Entry point. Runs a Pygame window at 1200×800 with a real-time fractal
rendered using Python multiprocessing (or CUDA on PC).

Controls:
  WASD / Arrow keys   Pan
  + / - / Scroll      Zoom
  M / J / B           Switch fractal (Mandelbrot / Julia / Burning Ship)
  C                   Cycle colour palette
  Space               Toggle Julia animation
  R                   Reset view to default
  P                   Run benchmark (sequential vs parallel timing)
  S                   Save screenshot to screenshots/
  H                   Toggle help overlay
  ESC                 Quit
"""

import os
import sys
import time
import threading
import pygame
import numpy as np

# Put project root on path so imports work from any CWD
sys.path.insert(0, os.path.dirname(__file__))

from backend import get_backend_label, BACKEND, NUM_WORKERS
from fractals import FRACTAL_KEYS, FRACTALS
from fractals.julia import get_animated_c
from renderer.colormap import ColormapManager
from renderer.hud import HUD
from benchmark import run_benchmark, format_hud_message

# ─── Window & render constants ────────────────────────────────────────────────
WIN_W, WIN_H    = 1200, 800
TARGET_FPS      = 60
MAX_ITER        = 128          # 128 is visually rich and ~2× faster than 256

# ─── Default view bounds per fractal ─────────────────────────────────────────
DEFAULT_BOUNDS = {
    "Mandelbrot":   (-2.5,  1.0,  -1.25, 1.25),
    "Julia":        (-2.0,  2.0,  -1.5,  1.5 ),
    "Burning Ship": (-2.5,  1.5,  -1.75, 0.75),
}

ZOOM_SPEED        = 0.15   # fraction of current range to zoom per step
PAN_SPEED         = 0.05   # fraction of current range to pan per keypress
JULIA_C_THRESHOLD = 0.002  # minimum c change before triggering a new Julia render


# ─── View state ──────────────────────────────────────────────────────────────

class ViewState:
    """Tracks the visible region of the complex plane."""

    def __init__(self, bounds):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds

    @property
    def bounds(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def zoom(self):
        """Zoom level relative to the full Mandelbrot extent (~3.5 units wide)."""
        return 3.5 / (self.xmax - self.xmin)

    @property
    def center(self):
        cx = (self.xmin + self.xmax) / 2
        cy = (self.ymin + self.ymax) / 2
        return cx, cy

    def zoom_in(self, factor=None):
        f = factor or ZOOM_SPEED
        dx = (self.xmax - self.xmin) * f / 2
        dy = (self.ymax - self.ymin) * f / 2
        self.xmin += dx; self.xmax -= dx
        self.ymin += dy; self.ymax -= dy

    def zoom_out(self, factor=None):
        f = factor or ZOOM_SPEED
        dx = (self.xmax - self.xmin) * f / 2
        dy = (self.ymax - self.ymin) * f / 2
        self.xmin -= dx; self.xmax += dx
        self.ymin -= dy; self.ymax += dy

    def pan(self, dx_frac, dy_frac):
        dx = (self.xmax - self.xmin) * dx_frac
        dy = (self.ymax - self.ymin) * dy_frac
        self.xmin += dx; self.xmax += dx
        self.ymin += dy; self.ymax += dy

    def zoom_to_pixel(self, px, py, screen_w, screen_h, direction):
        """Zoom centred on a specific pixel (mouse scroll)."""
        # Map pixel to complex plane
        mx = self.xmin + px * (self.xmax - self.xmin) / screen_w
        my = self.ymin + py * (self.ymax - self.ymin) / screen_h

        f = ZOOM_SPEED if direction > 0 else -ZOOM_SPEED
        # Shift centre toward/away from mouse point
        self.xmin = mx + (self.xmin - mx) * (1 - f)
        self.xmax = mx + (self.xmax - mx) * (1 - f)
        self.ymin = my + (self.ymin - my) * (1 - f)
        self.ymax = my + (self.ymax - my) * (1 - f)


# ─── Async render worker ──────────────────────────────────────────────────────

class RenderWorker:
    """
    Runs fractal rendering in a background thread so the UI stays responsive.
    The actual heavy computation happens inside multiprocessing Pool workers
    spawned by the fractal modules — this thread just orchestrates them.
    """

    def __init__(self):
        self._thread   = None
        self._result   = None
        self._running  = False
        self.render_ms = 0.0

    @property
    def is_busy(self):
        return self._running

    @property
    def result(self):
        r = self._result
        self._result = None
        return r

    def submit(self, fn, width, height, bounds, max_iter, extra_kwargs=None):
        """Start a new render in a background thread (drops if already running)."""
        if self._running:
            return  # skip — previous render still in progress
        self._running = True
        kwargs = extra_kwargs or {}

        def _work():
            t0 = time.perf_counter()
            arr = fn(width, height, bounds, max_iter, **kwargs)
            self.render_ms = (time.perf_counter() - t0) * 1000
            self._result  = arr
            self._running = False

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()


# ─── Main application ─────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("⚡ GPU Fractal Explorer  —  Parallel Processing Demo")
    clock    = pygame.time.Clock()

    # ── State ────────────────────────────────────────────────────────────
    fractal_idx   = 0                    # index into FRACTAL_KEYS
    fractal_name  = FRACTAL_KEYS[fractal_idx]
    view          = ViewState(DEFAULT_BOUNDS[fractal_name])
    colormap_mgr  = ColormapManager()
    hud           = HUD(WIN_W, WIN_H)
    worker        = RenderWorker()

    dirty         = True    # True = need to re-render
    julia_anim    = False
    anim_start    = time.time()
    _last_c       = None    # track last Julia c to avoid redundant submits

    # Current rendered surface (displayed while a new render is in flight)
    display_surf  = pygame.Surface((WIN_W, WIN_H))
    display_surf.fill((10, 10, 20))

    fps           = 0.0
    frame_count   = 0
    fps_timer     = time.time()

    os.makedirs("screenshots", exist_ok=True)

    print(f"\n  Fractal Explorer starting — backend: {get_backend_label()}")
    print(f"  Workers: {NUM_WORKERS}  |  Max iterations: {MAX_ITER}")
    print(f"  Press H for controls\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0

        # ── Events ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = event.key

                # Quit
                if key == pygame.K_ESCAPE:
                    running = False

                # Pan
                elif key in (pygame.K_LEFT,  pygame.K_a): view.pan(-PAN_SPEED, 0); dirty=True
                elif key in (pygame.K_RIGHT, pygame.K_d): view.pan( PAN_SPEED, 0); dirty=True
                elif key in (pygame.K_UP,    pygame.K_w): view.pan(0, -PAN_SPEED); dirty=True
                elif key in (pygame.K_DOWN,  pygame.K_s): view.pan(0,  PAN_SPEED); dirty=True

                # Zoom
                elif key in (pygame.K_PLUS,  pygame.K_EQUALS, pygame.K_KP_PLUS):
                    view.zoom_in(); dirty=True
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    view.zoom_out(); dirty=True

                # Switch fractal
                elif key == pygame.K_m:
                    fractal_name = "Mandelbrot"; fractal_idx = FRACTAL_KEYS.index(fractal_name)
                    view = ViewState(DEFAULT_BOUNDS[fractal_name]); dirty=True
                elif key == pygame.K_j:
                    fractal_name = "Julia"; fractal_idx = FRACTAL_KEYS.index(fractal_name)
                    view = ViewState(DEFAULT_BOUNDS[fractal_name]); dirty=True; julia_anim=True
                elif key == pygame.K_b:
                    fractal_name = "Burning Ship"; fractal_idx = FRACTAL_KEYS.index(fractal_name)
                    view = ViewState(DEFAULT_BOUNDS[fractal_name]); dirty=True

                # Palette
                elif key == pygame.K_c:
                    colormap_mgr.next(); dirty=True

                # Julia animation toggle
                elif key == pygame.K_SPACE:
                    julia_anim = not julia_anim
                    anim_start  = time.time()
                    dirty = True

                # Reset view
                elif key == pygame.K_r:
                    view = ViewState(DEFAULT_BOUNDS[fractal_name]); dirty=True

                # Benchmark
                elif key == pygame.K_p:
                    _run_benchmark_async(worker, hud, fractal_name, view, WIN_W, WIN_H)

                # Screenshot
                elif key == pygame.K_s:
                    _save_screenshot(display_surf)

                # Help toggle
                elif key == pygame.K_h:
                    hud.show_help = not hud.show_help

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                view.zoom_to_pixel(mx, my, WIN_W, WIN_H, event.y)
                dirty = True

        # ── Submit render ────────────────────────────────────────────────
        if not worker.is_busy:
            kwargs = {}
            should_render = dirty

            if fractal_name == "Julia":
                t  = time.time() - anim_start if julia_anim else 0
                new_c = get_animated_c(t) if julia_anim else complex(-0.7, 0.27015)
                kwargs["c"] = new_c

                # For Julia animation: only re-render when c has moved enough
                if julia_anim:
                    if _last_c is None or abs(new_c - _last_c) >= JULIA_C_THRESHOLD:
                        should_render = True
                        _last_c = new_c

            if should_render:
                dirty = False
                worker.submit(
                    FRACTALS[fractal_name],
                    WIN_W, WIN_H,
                    view.bounds,
                    MAX_ITER,
                    extra_kwargs=kwargs,
                )

        # ── Collect finished render ───────────────────────────────────────
        render_result = worker.result
        if render_result is not None:
            rgb = colormap_mgr.apply(render_result)
            # pygame surface expects (W, H, 3), numpy gives (H, W, 3) → transpose axes
            pygame_arr = np.transpose(rgb, (1, 0, 2))
            pygame.surfarray.blit_array(display_surf, pygame_arr)

        # ── Draw frame ───────────────────────────────────────────────────
        screen.blit(display_surf, (0, 0))

        # FPS counter (updated every 30 frames)
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_timer)
            fps_timer = time.time()

        cx, cy = view.center
        hud.draw(
            screen,
            fractal_name   = fractal_name,
            palette_name   = colormap_mgr.current_name,
            zoom           = view.zoom,
            cx=cx, cy=cy,
            fps            = fps,
            render_ms      = worker.render_ms,
            backend_label  = get_backend_label(),
            julia_animating= julia_anim and fractal_name == "Julia",
            max_iter       = MAX_ITER,
        )

        pygame.display.flip()

    pygame.quit()


# ─── Helpers ────────────────────────────────────────────────────────────────

def _run_benchmark_async(worker, hud, fractal_name, view, width, height):
    """Run benchmark in a background thread so the UI doesn't freeze."""
    render_fn = FRACTALS[fractal_name]

    def _bench():
        result = run_benchmark(
            width=width // 2,   # use half-res for speed
            height=height // 2,
            bounds=view.bounds,
            max_iter=MAX_ITER,
            fractal_render_fn=render_fn,
        )
        hud.benchmark_msg = format_hud_message(result)

    t = threading.Thread(target=_bench, daemon=True)
    t.start()


def _save_screenshot(surface):
    """Save current screen to screenshots/ folder with a timestamped name."""
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join("screenshots", f"fractal_{ts}.png")
    pygame.image.save(surface, path)
    print(f"  Screenshot saved → {path}")


if __name__ == "__main__":
    # Required on macOS/Windows for multiprocessing safety
    import multiprocessing
    multiprocessing.freeze_support()
    main()
