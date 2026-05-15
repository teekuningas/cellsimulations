# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Generate MP4 animation of ring atmosphere simulation.

Uses symmetric pressure + gravity model.

Run:  uv run cellsimulations/atmosphere/animate_ring.py
Then: nix shell nixpkgs#ffmpeg --command ffmpeg -framerate 30 -i cellsimulations/atmosphere/frames_ring/f_%04d.png -c:v libx264 -pix_fmt yuv420p cellsimulations/atmosphere/ring.mp4
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import RingGeometry, Params, step_ring, diagnostics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as clt

# -- Config --
N_LAYERS, N_CELLS = 8, 48
geo = RingGeometry(N_LAYERS, N_CELLS)
p = Params(solar=0.25, cooling=0.03, diffuse=0.0)
G = 0.08

SPHERE_R = 10.0
CELL_R = 0.5
TOTAL_FRAMES = 600
SUBSTEPS = 5
SUN_SPEED = 0.01
FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'frames_ring')

# -- State --
T = np.zeros(geo.shape)
u = np.zeros(geo.shape)
w = np.zeros(geo.shape)


def cell_xy(n_layers, n_cells, R, cr):
    pos = np.zeros((n_layers, n_cells, 2))
    for layer in range(n_layers):
        r = R + layer * cr * 2.2
        for cell in range(n_cells):
            a = cell / n_cells * 2 * np.pi
            pos[layer, cell] = [np.cos(a) * r, np.sin(a) * r]
    return pos

def make_dots(pos, cr):
    circles = []
    for layer in range(pos.shape[0]):
        for cell in range(pos.shape[1]):
            circles.append(patches.Circle(pos[layer, cell], radius=cr))
    return clt.PatchCollection(circles, edgecolors='gray', linewidths=0.2)

def make_arrows(pos, u_f, w_f, n_cells, scale=3.0):
    arrows = []
    for layer in range(pos.shape[0]):
        for cell in range(n_cells):
            x, y = pos[layer, cell]
            a = cell / n_cells * 2 * np.pi
            uv, wv = u_f[layer, cell] * scale, w_f[layer, cell] * scale
            dx = -np.sin(a) * uv + np.cos(a) * wv
            dy = np.cos(a) * uv + np.sin(a) * wv
            if abs(dx) + abs(dy) > 1e-6:
                arrows.append(patches.Arrow(x, y, dx, dy, width=0.3))
    if not arrows:
        arrows.append(patches.Arrow(0, 0, 0, 0, width=0))
    return clt.PatchCollection(arrows, color='steelblue', alpha=0.7)

def t_colors(T_field):
    flat = T_field.ravel()
    vmax = max(abs(flat.max()), abs(flat.min()), 0.01)
    return plt.cm.RdBu_r((flat + vmax) / (2 * vmax))

def sun_marker_xy(sun_angle, R, n_layers, cr):
    """Sun orbits outside the atmosphere, not inside the earth."""
    r = R + n_layers * cr * 2.2 + 2.5
    return np.cos(sun_angle) * r, np.sin(sun_angle) * r


# -- Run --
os.makedirs(FRAMES_DIR, exist_ok=True)
positions = cell_xy(N_LAYERS, N_CELLS, SPHERE_R, CELL_R)
sun = 0.0

print(f"Ring {N_LAYERS}x{N_CELLS}, generating {TOTAL_FRAMES} frames...")
for frame in range(TOTAL_FRAMES):
    for _ in range(SUBSTEPS):
        T, u, w = step_ring(T, u, w, geo, p, sun, g=G)
        sun += SUN_SPEED

    d = diagnostics(T, u, w)
    if frame % 100 == 0:
        print(f"  {frame:4d}/{TOTAL_FRAMES} | E={d['total']:8.1f} wind={d['max_wind']:.3f}")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal'); ax.axis('off')
    ext = SPHERE_R + N_LAYERS * CELL_R * 2.5 + 2
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
    ax.add_patch(plt.Circle((0,0), SPHERE_R, color='#4a86c8', alpha=0.15))

    # Sun marker
    sx, sy = sun_marker_xy(sun, SPHERE_R, N_LAYERS, CELL_R)
    ax.plot(sx, sy, 'o', color='gold', markersize=12, zorder=10)

    dots = make_dots(positions, CELL_R)
    dots.set_facecolors(t_colors(T))
    ax.add_collection(dots)
    ax.add_collection(make_arrows(positions, u, w, N_CELLS))

    ax.set_title(f'Ring Atmosphere — frame {frame}  E={d["total"]:.0f}  wind={d["max_wind"]:.2f}',
                 fontsize=11)
    fig.savefig(os.path.join(FRAMES_DIR, f'f_{frame:04d}.png'), dpi=80)
    plt.close(fig)

print(f"Done. {TOTAL_FRAMES} frames in {FRAMES_DIR}/")
print(f"To make MP4:")
print(f"  nix shell nixpkgs#ffmpeg --command ffmpeg -framerate 30 -i {FRAMES_DIR}/f_%04d.png -c:v libx264 -pix_fmt yuv420p cellsimulations/atmosphere/ring.mp4")
