# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Ring experiment — vertical cross-section convection.

Uses symmetric pressure + gravity. Buoyancy-like circulation emerges
from their competition without an explicit buoyancy rule.

Run:  uv run cellsimulations/atmosphere/run_ring.py
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
TOTAL_FRAMES = 500
SUBSTEPS = 5
SNAPSHOT_EVERY = 50
SUN_SPEED = 0.01
OUTPUT = os.path.join(os.path.dirname(__file__), 'output_ring')

# -- Initial state --
T = np.zeros(geo.shape)
u = np.zeros(geo.shape)
w = np.zeros(geo.shape)


# -- Viz helpers --
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


# -- Run --
os.makedirs(OUTPUT, exist_ok=True)
positions = cell_xy(N_LAYERS, N_CELLS, SPHERE_R, CELL_R)
sun = 0.0
energy_hist = []

print(f"Ring {N_LAYERS}x{N_CELLS}, {TOTAL_FRAMES} frames, params: {p}, g={G}")
for frame in range(TOTAL_FRAMES):
    for _ in range(SUBSTEPS):
        T, u, w = step_ring(T, u, w, geo, p, sun, g=G)
        sun += SUN_SPEED

    d = diagnostics(T, u, w)
    energy_hist.append(d['total'])

    if frame % 50 == 0:
        print(f"  {frame:4d} | E={d['total']:8.1f} wind={d['max_wind']:.3f} T={d['mean_T']:.3f}")

    if frame % SNAPSHOT_EVERY == 0 or frame == TOTAL_FRAMES - 1:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect('equal'); ax.axis('off')
        ext = SPHERE_R + N_LAYERS * CELL_R * 2.5 + 2
        ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
        ax.add_patch(plt.Circle((0,0), SPHERE_R, color='#4a86c8', alpha=0.2))

        dots = make_dots(positions, CELL_R)
        dots.set_facecolors(t_colors(T))
        ax.add_collection(dots)
        ax.add_collection(make_arrows(positions, u, w, N_CELLS))

        ax.set_title(f'Ring -- frame {frame}  E={d["total"]:.0f}  wind={d["max_wind"]:.2f}')
        fig.savefig(os.path.join(OUTPUT, f'ring_{frame:04d}.png'), dpi=100)
        plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(energy_hist, color='steelblue')
ax.set(xlabel='Frame', ylabel='Energy', title='Total energy')
ax.grid(True, alpha=0.3); plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, 'energy.png'), dpi=100)
plt.close(fig)
print("Done.")
