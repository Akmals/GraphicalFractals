"""
backend.py — Backend selector for fractal rendering.

Change BACKEND to switch between rendering modes:
  "SEQUENTIAL"       — single-threaded NumPy (slow, for benchmarking)
  "MULTIPROCESSING"  — Python multiprocessing Pool (parallel CPU, Mac)
  "CUDA"             — Numba CUDA kernels (GPU, PC with RTX 4060 Ti)
"""

import os

# ─── CHANGE THIS TO "CUDA" WHEN ON PC ───────────────────────────────────────
BACKEND = "MULTIPROCESSING"
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
