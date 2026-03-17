"""
hud.py — On-screen HUD (heads-up display) overlay.

Draws text overlays onto the Pygame surface using pygame.font.
All rendering is done directly on the surface passed in — no separate surface needed.
"""

import pygame


# ─── Colours ─────────────────────────────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0  )
YELLOW     = (255, 220, 50 )
CYAN       = (80,  220, 220)
GREEN      = (80,  240, 100)
ORANGE     = (255, 160, 40 )
DIM        = (180, 180, 180)
BG_OVERLAY = (0,   0,   0,  160)   # semi-transparent (used with alpha surface)


class HUD:
    def __init__(self, screen_w: int, screen_h: int):
        pygame.font.init()
        self._w = screen_w
        self._h = screen_h

        self._font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 14)
        self._font_xs = pygame.font.SysFont("monospace", 12)

        self.show_help     = False
        self.benchmark_msg = None   # set after benchmark runs

        self._CONTROLS = [
            ("WASD / Arrows", "Pan"),
            ("+ / -  / Scroll", "Zoom"),
            ("M",  "Mandelbrot"),
            ("J",  "Julia Set"),
            ("B",  "Burning Ship"),
            ("C",  "Cycle Palette"),
            ("Space", "Animate Julia"),
            ("R",  "Reset View"),
            ("P",  "Benchmark"),
            ("S",  "Screenshot"),
            ("H",  "Toggle Help"),
            ("ESC","Quit"),
        ]

    def _blit_text(self, surface, text, pos, font, color=WHITE, shadow=True):
        if shadow:
            shadow_surf = font.render(text, True, BLACK)
            surface.blit(shadow_surf, (pos[0]+1, pos[1]+1))
        surf = font.render(text, True, color)
        surface.blit(surf, pos)

    def draw(self, surface, *, fractal_name, palette_name, zoom,
             cx, cy, fps, render_ms, backend_label,
             julia_animating=False, max_iter=256):
        """Draw all HUD elements onto the given surface."""

        # ── Top-left info panel ───────────────────────────────────────────
        lines = [
            (f"FPS: {fps:.0f}",                         YELLOW),
            (f"Fractal: {fractal_name}",                GREEN),
            (f"Palette: {palette_name}",                CYAN),
            (f"Backend: {backend_label}",               ORANGE),
            (f"Render:  {render_ms:.0f} ms",            DIM),
            (f"Iter:    {max_iter}",                    DIM),
            (f"Zoom:    {zoom:.2e}",                    WHITE),
            (f"Center:  ({cx:.6f}, {cy:.6f})",          WHITE),
        ]

        if julia_animating:
            lines.append(("● Julia Animating",          (255, 100, 100)))

        pad = 10
        line_h = 20
        for i, (text, color) in enumerate(lines):
            self._blit_text(surface, text, (pad, pad + i * line_h),
                            self._font_sm, color)

        # ── Benchmark overlay ─────────────────────────────────────────────
        if self.benchmark_msg:
            bw, bh = 520, 80
            bx = (self._w - bw) // 2
            by = self._h - bh - 20
            box = pygame.Surface((bw, bh), pygame.SRCALPHA)
            box.fill((0, 0, 0, 180))
            surface.blit(box, (bx, by))
            pygame.draw.rect(surface, YELLOW, (bx, by, bw, bh), 2, border_radius=6)
            for i, line in enumerate(self.benchmark_msg.split("\n")):
                self._blit_text(surface, line, (bx + 14, by + 10 + i * 24),
                                self._font_lg, YELLOW, shadow=True)

        # ── Controls help panel (H key) ───────────────────────────────────
        if self.show_help:
            cols = 2
            rows = (len(self._CONTROLS) + cols - 1) // cols
            pw, ph = 360, rows * 22 + 30
            px = (self._w - pw) // 2
            py = (self._h - ph) // 2

            panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
            panel.fill((10, 10, 30, 210))
            surface.blit(panel, (px, py))
            pygame.draw.rect(surface, CYAN, (px, py, pw, ph), 2, border_radius=8)

            self._blit_text(surface, "── Controls ──",
                            (px + pw//2 - 55, py + 6), self._font_sm, CYAN)

            col_w = pw // cols
            for i, (key, action) in enumerate(self._CONTROLS):
                col = i // rows
                row = i % rows
                x = px + col * col_w + 14
                y = py + 28 + row * 22
                self._blit_text(surface, f"{key:<16} {action}",
                                (x, y), self._font_xs, WHITE, shadow=False)

        # ── Bottom-right hint ─────────────────────────────────────────────
        hint = "H = Help"
        hint_surf = self._font_xs.render(hint, True, DIM)
        surface.blit(hint_surf, (self._w - hint_surf.get_width() - 10,
                                  self._h - hint_surf.get_height() - 8))
