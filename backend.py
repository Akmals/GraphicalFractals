"""
backend.py — Backend selector for fractal rendering.

Change BACKEND to switch between rendering modes:
  "SEQUENTIAL"       — single-threaded NumPy (slow, for benchmarking)
  "MULTIPROCESSING"  — Numba parallel CPU JIT (current Mac path)
  "CUDA"             — Numba CUDA GPU kernels (PC with RTX 4060 Ti)

You can also override without editing this file by setting the env var
FRACTAL_BACKEND, e.g.  FRACTAL_BACKEND=CUDA python main.py
"""

import os

# ─── CHANGE THIS TO "CUDA" WHEN ON PC ───────────────────────────────────────
BACKEND = os.environ.get("FRACTAL_BACKEND", "MULTIPROCESSING").upper()
# ────────────────────────────────────────────────────────────────────────────

# Auto-detect number of CPU cores for multiprocessing
NUM_WORKERS = os.cpu_count() or 4

def get_backend_label() -> str:
    labels = {
        "SEQUENTIAL": "CPU Sequential",
        "MULTIPROCESSING": f"CPU Parallel ({NUM_WORKERS} cores)",
        "CUDA": "GPU CUDA (RTX 4060 Ti)",
    }
    return labels.get(BACKEND, BACKEND)
