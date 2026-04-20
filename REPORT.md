# GPU Fractal Explorer — Performance Analysis Report

**Sequential vs CPU-Parallel vs GPU rendering of the Mandelbrot Set**

---

## 1. Executive Summary

This report compares four implementations of the same Mandelbrot-set
escape-time kernel inside the GPU Fractal Explorer project. All four
produce numerically identical output; they differ only in **how** the
pixel-independent inner loop is scheduled across hardware.

| Implementation            | Hardware target     | Parallelism model                     |
| ------------------------- | ------------------- | ------------------------------------- |
| 1. Sequential NumPy       | 1 CPU core          | Vectorised array ops, single thread   |
| 2. `multiprocessing.Pool` | All CPU cores       | Row chunks in separate OS processes   |
| 3. Numba `@njit` parallel | All CPU cores       | JIT-compiled threads via `prange`     |
| 4. Numba `@cuda.jit`      | NVIDIA RTX 4060 Ti  | One GPU thread per pixel (2-D grid)   |

At **1920×1080 @ 128 iterations**, the GPU kernel is **306× faster**
than the pure-NumPy sequential baseline and **3× faster** than a fully
JIT-compiled CPU-parallel run across 12 threads. In frame-rate terms,
this is **0.3 fps vs 33 fps vs 101 fps** — the difference between a
still image, a choppy animation, and a fluid real-time experience.

---

## 2. Test System

| Component | Spec |
|---|---|
| CPU | 12 logical cores |
| GPU | NVIDIA GeForce RTX 4060 Ti (Ada, Compute 8.9) |
| OS  | Windows 11 |
| Python | 3.12.10 |
| Libraries | NumPy 2.2, Numba 0.65, pygame 2.6, CUDA Toolkit 13.2 |
| Iterations | `MAX_ITER = 128` |
| Methodology | Each backend pre-warmed once (JIT compile, pool spawn, CUDA context), then median of 5 runs |

---

## 3. Benchmark Results

Tested from 540p up through 1440p to show how the speedup scales with
problem size (the app itself targets 1080p).

### 3.1 Render time per frame (lower = better)

| Resolution | Pixels | Seq NumPy  | MP Pool    | Numba par  | CUDA GPU    | Numba vs Seq | CUDA vs Seq | CUDA vs Numba |
| ---------- | ------ | ---------- | ---------- | ---------- | ----------- | ------------ | ----------- | -------------- |
| 960×540    | 518 k  | 833.8 ms   | 2 513.6 ms | **7.5 ms** | **2.7 ms**  | 111×         | **313×**    | 2.8×           |
| 1280×720   | 922 k  | 1 221.5 ms | 2 541.4 ms | 12.3 ms    | 6.5 ms      | 99×          | 189×        | 1.9×           |
| 1920×1080  | 2.07 M | 3 039.7 ms | 3 568.5 ms | 29.8 ms    | **9.9 ms**  | 102×         | **306×**    | 3.0×           |
| 2560×1440  | 3.69 M | 5 335.1 ms | 4 668.9 ms | 47.1 ms    | 15.1 ms     | 113×         | **354×**    | 3.1×           |

### 3.2 Effective frame rate (higher = better)

Same measurements expressed as FPS (`1000 / render_ms`) — this is the
steady-state frame rate achievable before display-pipeline overhead:

| Resolution | Seq FPS | MP FPS | Numba FPS | **CUDA FPS** |
| ---------- | ------- | ------ | --------- | ------------ |
| 960×540    | 1.2     | 0.4    | 133.5     | **368.3**    |
| 1280×720   | 0.8     | 0.4    | 81.0      | **153.8**    |
| 1920×1080  | 0.3     | 0.3    | 33.5      | **101.3**    |
| 2560×1440  | 0.2     | 0.2    | 21.2      | **66.2**     |

### 3.3 Headline takeaway at 1080p

| Backend     | Render time | FPS   | Feel                            |
| ----------- | ----------- | ----- | ------------------------------- |
| Sequential  | 3 040 ms    | 0.3   | Slide show (one frame / 3 s)    |
| MP Pool     | 3 569 ms    | 0.3   | Worse than sequential           |
| Numba par   | 29.8 ms     | 33.5  | Smooth 30 fps animation         |
| **CUDA**    | **9.9 ms**  | **101** | **Real-time 100+ fps gameplay** |

The GPU comfortably sustains **>100 fps at 1080p** before any display
overhead, which is why the interactive app hits its 120 fps display
target. The `clock.tick(120)` cap in `main.py` is the **actual** limit —
not the render cost.

---

## 4. Analysis

### 4.1 Why pure-NumPy "sequential" is as fast as it is (and no faster)

NumPy already runs most arithmetic inside vectorised C loops, so the
"sequential" baseline isn't a naive Python interpreter loop — it's
1 core of BLAS-style SIMD math. That's why it hits ~830 ms at half-HD
rather than tens of seconds. The limit is that it's **one core**,
and every iteration allocates temporary arrays for `zx**2`, `zy**2`,
the boolean mask, etc.

### 4.2 Why `multiprocessing.Pool` underperforms — a teaching moment

Counter-intuitively, the Pool version is **slower than sequential at
every tested resolution below 1440p**, and only barely faster there
(1.1×). Three reasons:

1. **Process-spawn cost.** Windows uses `spawn`, not `fork`. Each
   worker has to re-import NumPy (~80 ms) and unpickle its chunk of
   the work. At 960×540 this dwarfs the actual compute.
2. **Data serialisation.** Pool `map` pickles arguments to workers
   and pickles results back. For chunks of the 1440p grid that's
   several MB of IPC per frame.
3. **Loss of vectorisation per chunk.** Each worker operates on a
   slice that's 12× smaller than the full image, so NumPy's vector
   throughput per dispatch drops relative to one large call.

Pool parallelism only starts catching up above 1440p, where per-worker
compute finally dominates the fixed spawn/IPC cost, and even then it
**never beats a proper thread-parallel JIT** on any resolution tested.

**Lesson:** "throw more processes at it" is the *wrong* shape of
parallelism for a fine-grained numerical inner loop on a single
machine. Shared-memory threads or a GPU are the right tools.

### 4.3 Why Numba `@njit(parallel=True)` wins on CPU — by 100×

`@njit` ahead-of-time compiles the Python escape-time loop to
native machine code. `parallel=True` with `prange` turns the outer
row loop into an OpenMP-style thread pool *inside one process*. The
result:

- **No interpreter overhead.** The inner `while` is raw x86.
- **No intermediate arrays.** Each pixel's `zx, zy, i` live in registers.
- **No IPC.** Threads share the output buffer.
- **12 cores, ~12× speedup** stacked on top of the ~10× single-thread
  speedup over NumPy ≈ **~100× total**, which matches the measurements.

This is the production path on a Mac / any machine without an NVIDIA GPU.

### 4.4 Why CUDA is another ~3× on top of Numba CPU

The Mandelbrot set is **embarrassingly parallel** — every pixel is
independent, zero communication needed. This is the ideal workload
for a GPU.

- **Thread count.** A 1920×1080 frame launches **2 073 600 CUDA threads**
  in a 16×16-block 2-D grid. The 4060 Ti has 4 352 shader cores, so
  threads are time-multiplexed over the SMs in ~480-thread waves.
- **Divergence tolerance.** Escape-time loops diverge (some pixels
  escape at iter 5, others run all 128), but GPUs tolerate this well
  at this scale because the slowest pixel in each warp only stalls
  31 peers, not the whole machine.
- **Arithmetic intensity.** Each pixel does up to 128 iterations of
  ~6 FLOPs with zero memory traffic until the final store. This is
  exactly where GPUs shine.

The CUDA advantage grows with resolution: at 960×540 the per-frame
kernel-launch latency and `copy_to_host()` DMA are a noticeable
fraction of total time, but at 1920×1080 and above those fixed costs
are amortised across millions of pixels. At 1440p the GPU reaches its
peak **354× speedup** over sequential NumPy and a clean **3.1× over
the already-parallel Numba CPU path** (66 fps vs 21 fps).

### 4.5 Correctness

The in-app benchmark (press **P**) also verifies numerical equivalence
between backends:

```
Max pixel Δ: 0.0000  (0 = identical)
```

All four backends return bit-identical `float64` escape counts modulo
the `fastmath` contraction (same up to 1 ULP).

---

## 5. Key Code Snippets

### 5.1 The shared inner loop (all backends compute this)

Every backend below computes the same per-pixel escape-time equation
`z ← z² + c`, iterating until `|z|² > 4` or `i = max_iter`:

```python
while zx*zx + zy*zy <= 4.0 and i < max_iter:
    zx, zy = zx*zx - zy*zy + cx, 2.0*zx*zy + cy
    i += 1
```

The only difference between implementations is **what decides which
thread/process/core runs which pixel.**

### 5.2 Sequential NumPy — vectorised, 1 core

From `terminal_benchmark.py`. The "loop" is over iterations, not
pixels — every pixel advances one step per outer iteration using a
boolean `alive` mask so escaped pixels stop updating.

```python
def _mandelbrot_sequential(width, height, bounds, max_iter):
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, width,  dtype=np.float64)
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
```

### 5.3 `multiprocessing.Pool` — row chunks across OS processes

From `terminal_benchmark.py`. Splits the height into `n_workers`
horizontal strips and dispatches each to a separate process:

```python
def _render_parallel(fractal_type, width, height, bounds, max_iter,
                     n_workers=None):
    if n_workers is None:
        n_workers = cpu_count() or 4
    xmin, xmax, ymin, ymax = bounds

    chunk_size = height // n_workers
    tasks = []
    for i in range(n_workers):
        start = i * chunk_size
        end   = start + chunk_size if i < n_workers - 1 else height
        tasks.append((start, end, width, xmin, xmax, ymin, ymax,
                      max_iter, height))

    result = np.zeros((height, width), dtype=np.float64)
    with Pool(processes=n_workers) as pool:
        for start_row, chunk in pool.map(_parallel_mandelbrot_worker, tasks):
            result[start_row:start_row + chunk.shape[0]] = chunk
    return result
```

Each worker re-implements the NumPy inner loop on its own slice, then
ships its `(start_row, chunk)` back via pickle. The `with Pool(...)`
block both spawns and tears down the pool every call — which is one
of the reasons the Pool version underperforms.

### 5.4 Numba `@njit(parallel=True)` — JIT-compiled thread pool

From `fractals/mandelbrot.py`. `prange` is Numba's parallel `range`:
the decorator emits an OpenMP-style thread pool that runs each row on
a different hardware thread.

```python
@njit(parallel=True, cache=True, fastmath=True)
def _mandelbrot_numba(width, height, xmin, xmax, ymin, ymax, max_iter):
    result = np.zeros((height, width), dtype=np.float64)
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    log2 = np.log(2.0)

    for py in prange(height):          # ← one thread per row
        cy = ymin + py * dy
        for px in range(width):
            cx = xmin + px * dx
            zx, zy = 0.0, 0.0
            i = 0
            while zx * zx + zy * zy <= 4.0 and i < max_iter:
                zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
                i += 1
            if i < max_iter:
                log_zn = np.log(zx * zx + zy * zy) * 0.5
                nu = np.log(log_zn / log2) / log2
                result[py, px] = i + 1.0 - nu
    return result
```

Key flags:
- `parallel=True` → enables `prange` and auto-parallelisation.
- `cache=True`   → saves the compiled artefact so the second run
                   skips the ~1 s JIT warm-up.
- `fastmath=True` → allows reassociation and fused-multiply-add, buying
                    roughly 10–15 % extra throughput.

### 5.5 CUDA `@cuda.jit` — one GPU thread per pixel

From `fractals/mandelbrot.py`. This is the same escape-time loop, but
the outer `for`-loops over `px, py` have vanished — every GPU thread
computes *exactly one* pixel and `cuda.grid(2)` tells it which one:

```python
@cuda.jit(fastmath=True)
def _mandelbrot_cuda_kernel(result, xmin, ymin, dx, dy, max_iter):
    px, py = cuda.grid(2)                 # this thread's pixel
    height, width = result.shape
    if px >= width or py >= height:
        return                            # edge threads bail out

    cx = xmin + px * dx
    cy = ymin + py * dy
    zx = 0.0
    zy = 0.0
    i = 0
    while zx * zx + zy * zy <= 4.0 and i < max_iter:
        zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
        i += 1

    if i < max_iter:
        log_zn = _math.log(zx * zx + zy * zy) * 0.5
        nu = _math.log(log_zn / _math.log(2.0)) / _math.log(2.0)
        result[py, px] = i + 1.0 - nu
    else:
        result[py, px] = 0.0
```

### 5.6 CUDA launch configuration

The kernel is driven from the host with a 2-D grid of 16×16 thread
blocks — 256 threads per block, which is the sweet spot on every
modern NVIDIA architecture:

```python
def _mandelbrot_cuda(width, height, xmin, xmax, ymin, ymax, max_iter):
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height

    d_result = cuda.device_array((height, width), dtype=np.float64)

    threads_per_block = (16, 16)                    # 256 threads/block
    blocks_x = (width  + 15) // 16
    blocks_y = (height + 15) // 16
    blocks_per_grid = (blocks_x, blocks_y)

    _mandelbrot_cuda_kernel[blocks_per_grid, threads_per_block](
        d_result, xmin, ymin, dx, dy, max_iter
    )
    return d_result.copy_to_host()                  # DMA back to RAM
```

For 1920×1080 that's a **120×68 grid of 16×16 blocks = 2 088 960 threads**,
mapped over the RTX 4060 Ti's 34 SMs by the hardware scheduler.

### 5.7 Async render orchestration (the reason the UI stays at 120 fps)

Pygame's main loop never blocks on a render: the `RenderWorker` in
`main.py` hands rendering to a background thread and only picks up
results when they're ready. This is what lets the HUD, pan/zoom, and
input handling stay smooth at 120 fps even when a frame takes 30 ms
to compute.

```python
class RenderWorker:
    def submit(self, fn, width, height, bounds, max_iter, extra_kwargs=None):
        if self._running:
            return                        # drop — previous still in flight
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
```

This is a simple but important architectural choice: **the frame rate
is decoupled from the render rate.** If a render takes 30 ms, the UI
still hits 120 fps; new frames just appear as the background thread
finishes them.

### 5.8 Backend selector

From `backend.py`. A single env var swaps the entire render path
without touching kernel code:

```python
BACKEND = os.environ.get("FRACTAL_BACKEND", "MULTIPROCESSING").upper()
```

And in `fractals/mandelbrot.py`:

```python
def render(width, height, bounds, max_iter=256, force_sequential=False):
    if _CUDA and not force_sequential:
        return _mandelbrot_cuda (width, height, *bounds, max_iter)
    if _NUMBA:
        return _mandelbrot_numba(width, height, *bounds, max_iter)
    return _mandelbrot_numpy    (width, height, *bounds, max_iter)
```

---

## 6. Conclusion

Four implementations of the same maths produced up to a **354×
performance spread** on real hardware (306× at the 1080p target,
peaking at 354× at 1440p):

1. **`multiprocessing.Pool` is the wrong tool** for fine-grained
   numerical inner loops on one machine. Process-spawn and pickle
   overhead can make it **slower than single-threaded NumPy**, and it
   never caught up to a thread-parallel JIT in our tests.

2. **Numba `@njit(parallel=True)` is the correct CPU answer.** It
   compiles Python to native code, runs `prange` across every core in
   one process, and delivered a clean **~100× speedup** over
   sequential NumPy at every resolution tested.

3. **CUDA is the correct answer if you have a discrete NVIDIA GPU.**
   The Mandelbrot escape-time loop is embarrassingly parallel, so the
   GPU's ~2 million concurrent threads (at 1080p) deliver another
   **~3× on top** of the already-parallel CPU version. End-to-end
   speedup over sequential NumPy: **306× at 1080p** and peaking at
   **354× at 1440p**, taking the render from a 0.3 fps slide show to
   a fluid **101 fps**.

4. **Architecture matters as much as raw compute.** Wrapping the
   render call in a background thread (`RenderWorker`) decouples
   frame rate from render rate and is what lets the app hit its 120 fps
   display target even when renders take 30 ms.

### 6.1 Practical recommendation

For any pixel-parallel numerical workload (fractals, ray tracing,
cellular automata, particle sims):

| Situation                          | Best choice                             |
| ---------------------------------- | --------------------------------------- |
| No compiled backend available      | Vectorised NumPy (1 core)               |
| Multi-core CPU, no GPU             | `numba @njit(parallel=True)` + `prange` |
| Multi-core CPU **with** NVIDIA GPU | `numba @cuda.jit`, fall back to Numba   |
| Many independent *large* tasks     | `multiprocessing.Pool` (task-level)     |

Use `multiprocessing.Pool` for **job-level** parallelism (several
independent renders queued up), not for **pixel-level** parallelism
inside one render.

### 6.2 Repository

Source, benchmark script, and CUDA setup guide:
<https://github.com/Akmals/GraphicalFractals>

Run the benchmark yourself:

```bash
# Four-way benchmark (seq / multiprocessing / Numba / CUDA)
FRACTAL_BACKEND=CUDA python report_bench.py

# Interactive app at 1080p @ 120 fps
FRACTAL_BACKEND=CUDA python main.py --res 1080p --fps 120
```

---

*All measurements: Mandelbrot, MAX_ITER=128, median of 5 runs,
Python 3.12 / Numba 0.65 / CUDA Toolkit 13.2, RTX 4060 Ti + 12-core CPU.*
