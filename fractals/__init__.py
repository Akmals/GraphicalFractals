from .mandelbrot import render as render_mandelbrot
from .julia import render as render_julia
from .burning_ship import render as render_burning_ship

FRACTALS = {
    "Mandelbrot": render_mandelbrot,
    "Julia":      render_julia,
    "Burning Ship": render_burning_ship,
}
FRACTAL_KEYS = list(FRACTALS.keys())
