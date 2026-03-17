"""
colormap.py — Color palette management.

Converts a 2D array of float iteration values into an RGB image (numpy array).
Uses matplotlib colormaps as lookup tables (LUTs) for rich, perceptually
uniform color palettes.

Each colormap is pre-baked into a 256-entry LUT at startup for fast mapping.
"""

import numpy as np
import matplotlib.pyplot as plt


class ColormapManager:
    # 12 curated palettes — cycle with C key
    PALETTE_NAMES = [
        "inferno", "plasma", "viridis", "magma",
        "hot",     "cool",   "twilight", "hsv",
        "YlOrRd",  "ocean",  "cubehelix", "gist_rainbow",
    ]

    DISPLAY_NAMES = [
        "Inferno", "Plasma", "Viridis", "Magma",
        "Hot",     "Arctic", "Twilight", "Rainbow",
        "Fire",    "Ocean",  "Cube",    "Spectrum",
    ]

    def __init__(self):
        self._index = 0
        self._luts = {}
        self._precompute_luts()

    def _precompute_luts(self):
        """Pre-bake all 256-entry RGB LUTs at startup."""
        for name in self.PALETTE_NAMES:
            cmap = plt.get_cmap(name)
            indices = np.linspace(0, 1, 256)
            # cmap returns RGBA floats in [0,1]; convert to uint8 RGB
            rgba = cmap(indices)
            self._luts[name] = (rgba[:, :3] * 255).astype(np.uint8)

    @property
    def current_name(self) -> str:
        return self.DISPLAY_NAMES[self._index]

    def next(self):
        """Advance to next palette."""
        self._index = (self._index + 1) % len(self.PALETTE_NAMES)

    def apply(self, iteration_array: np.ndarray) -> np.ndarray:
        """
        Map a 2D float array of iteration counts → RGB image (H×W×3 uint8).

        Steps:
          1. Normalise to [0, 1] (skip zeros which represent interior points)
          2. Scale to LUT index [0, 255]
          3. Map each index to an RGB triple from the LUT
          4. Set interior pixels (iteration=0) to pure black
        """
        lut = self._luts[self.PALETTE_NAMES[self._index]]

        # Normalise only non-zero values
        mask = iteration_array > 0
        result = np.zeros((*iteration_array.shape, 3), dtype=np.uint8)

        if mask.any():
            arr = iteration_array.copy()
            vmin = arr[mask].min()
            vmax = arr[mask].max()
            if vmax > vmin:
                arr[mask] = (arr[mask] - vmin) / (vmax - vmin)
            else:
                arr[mask] = 0.5

            indices = (arr * 255).astype(np.uint8)
            result = lut[indices]       # vectorised lookup — very fast
            result[~mask] = [0, 0, 0]  # interior → black

        return result
