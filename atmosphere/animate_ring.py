# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Animated ring model — slow sun reveals the Hadley convection cell.

Left:  Ring cross-section (temperature + wind arrows)
Right: Wind profile at 4 positions relative to sun

Run:  uv run cellsimulations/atmosphere/animate_ring.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import RingGeometry, Params, step_ring

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as clt

# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

N_LAYERS = 16
N_CELLS = 48
G = 0.04
SUN_SPEED = 0.002

FPS = 10
DURATION = 10
TOTAL_FRAMES = FPS * DURATION  # 100
SUBSTEPS = 20

SPHERE_R = 8.0
CELL_R = 0.38
FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'output_animations', 'ring')

p = Params(solar=0.20, cooling=0.025, c_sq=0.15, diffuse=0.0, drag=0.02)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def cell_positions(n_layers, n_cells, R, cr):
    pos = np.zeros((n_layers, n_cells, 2))
    for layer in range(n_layers):
        r = R + layer * cr * 2.2
        for cell in range(n_cells):
            angle = cell / n_cells * 2 * np.pi
            pos[layer, cell] = [np.cos(angle) * r, np.sin(angle) * r]
    return pos


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    geo = RingGeometry(N_LAYERS, N_CELLS)
    T = np.zeros(geo.shape)
    u = np.zeros(geo.shape)
    w = np.zeros(geo.shape)
    sun = 0.0

    os.makedirs(FRAMES_DIR, exist_ok=True)
    pos = cell_positions(N_LAYERS, N_CELLS, SPHERE_R, CELL_R)

    # Warm up
    print("Warming up...")
    for _ in range(400):
        T, u, w = step_ring(T, u, w, geo, p, sun, g=G)
        sun += SUN_SPEED

    print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps → {DURATION}s")
    ext = SPHERE_R + N_LAYERS * CELL_R * 2.5 + 3

    for frame in range(TOTAL_FRAMES):
        for _ in range(SUBSTEPS):
            T, u, w = step_ring(T, u, w, geo, p, sun, g=G)
            sun += SUN_SPEED

        sun_angle = sun % (2 * np.pi)

        # --- Create fresh figure each frame (avoids text ghosting) ---
        fig = plt.figure(figsize=(14, 7), facecolor='#131822')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.08)
        ax_ring = fig.add_subplot(gs[0])
        ax_prof = fig.add_subplot(gs[1])

        # === Left: Ring ===
        ax_ring.set_facecolor('#131822')
        ax_ring.set_aspect('equal')
        ax_ring.axis('off')
        ax_ring.set_xlim(-ext, ext)
        ax_ring.set_ylim(-ext, ext)

        # Planet core
        ax_ring.add_patch(plt.Circle((0, 0), SPHERE_R * 0.92,
                                     color='#1e2a3a', alpha=0.7))

        # Sun glow + marker
        beam_r = ext - 0.5
        sx = np.cos(sun_angle) * beam_r
        sy = np.sin(sun_angle) * beam_r
        ax_ring.add_patch(plt.Circle((sx, sy), 1.5, color='#FFD700', alpha=0.08))
        ax_ring.plot(sx, sy, '*', color='#FFD700', markersize=22, zorder=10)

        # Temperature cells (warm colormap, brighter)
        vmax = max(np.max(T), 0.5)
        norm_T = np.clip(T.ravel() / vmax, 0, 1)
        # Use a custom blend: cold=slate blue, hot=bright orange
        colors = plt.cm.YlOrRd(norm_T * 0.85 + 0.1)

        circles = []
        for layer in range(N_LAYERS):
            for cell in range(N_CELLS):
                circles.append(patches.Circle(pos[layer, cell], radius=CELL_R))
        dots = clt.PatchCollection(circles, edgecolors='#3a3a3a', linewidths=0.15)
        dots.set_facecolors(colors)
        ax_ring.add_collection(dots)

        # Wind arrows — bigger, more visible, on every layer but every 4th cell
        scale = 4.0
        for layer in range(N_LAYERS):
            for cell in range(0, N_CELLS, 4):
                x, y = pos[layer, cell]
                angle = cell / N_CELLS * 2 * np.pi
                uv = u[layer, cell] * scale
                wv = w[layer, cell] * scale
                dx = -np.sin(angle) * uv + np.cos(angle) * wv
                dy = np.cos(angle) * uv + np.sin(angle) * wv
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 0.08:
                    alpha = min(0.4 + mag * 0.8, 0.95)
                    ax_ring.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                                    arrowprops=dict(arrowstyle='->', lw=1.0,
                                                   color='#88ddff',
                                                   alpha=alpha,
                                                   mutation_scale=10))

        ax_ring.set_title('Ring Model — Hadley Convection Cell',
                         color='white', fontsize=12, pad=8)

        # === Right: Wind profile ===
        ax_prof.set_facecolor('#131822')

        offsets = [0, N_CELLS // 4, N_CELLS // 2, 3 * N_CELLS // 4]
        labels = ['☀ Under sun', '90° ahead', '● Opposite', '90° behind']
        line_colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1']

        for off, label, col in zip(offsets, labels, line_colors):
            col_idx = int((sun_angle / (2 * np.pi) * N_CELLS + off) % N_CELLS)
            ax_prof.plot(u[:, col_idx], range(N_LAYERS), color=col,
                        lw=2.5, label=label, alpha=0.85)

        ax_prof.axvline(0, color='#555', lw=0.8, ls='--')
        ax_prof.set_xlabel('Horizontal wind (u)', color='#ccc', fontsize=10)
        ax_prof.set_ylabel('Altitude (layer)', color='#ccc', fontsize=10)
        ax_prof.set_title('Wind Profile', color='white', fontsize=11, pad=8)
        ax_prof.set_xlim(-0.65, 0.65)
        ax_prof.set_ylim(0, N_LAYERS - 1)
        ax_prof.tick_params(colors='#999')
        for spine in ax_prof.spines.values():
            spine.set_color('#444')
        ax_prof.grid(True, alpha=0.12, color='#555')

        # Legend at bottom-right, below the curves
        ax_prof.legend(fontsize=9, loc='lower right', framealpha=0.3,
                      labelcolor='white', facecolor='#1a1a2e',
                      edgecolor='#444')

        # Frame info
        fig.text(0.5, 0.01,
                 f'frame {frame}/{TOTAL_FRAMES}  •  '
                 f'sun: {np.degrees(sun_angle):.0f}°  •  '
                 f'max |u|: {np.max(np.abs(u)):.2f}',
                 color='#666', fontsize=8, ha='center')

        fig.savefig(os.path.join(FRAMES_DIR, f'f_{frame:04d}.png'), dpi=100,
                   facecolor=fig.get_facecolor())
        plt.close(fig)

        if frame % 25 == 0:
            print(f"  {frame}/{TOTAL_FRAMES} ({100*frame//TOTAL_FRAMES}%)")

    print(f"\nFrames → {FRAMES_DIR}/")
    print(f"Stitch: nix shell nixpkgs#ffmpeg --command ffmpeg -framerate {FPS} "
          f"-i {FRAMES_DIR}/f_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p -crf 20 "
          f"{os.path.dirname(FRAMES_DIR)}/ring.mp4")


if __name__ == '__main__':
    main()
