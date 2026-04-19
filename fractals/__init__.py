from .mandelbrot import render as render_mandelbrot
from .julia import render as render_julia

FRACTALS = {
    "Mandelbrot": render_mandelbrot,
    "Julia":      render_julia,
}
FRACTAL_KEYS = list(FRACTALS.keys())
