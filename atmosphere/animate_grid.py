# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Animated grid model — Coriolis creates jet streams.

Three panels:
  Top:          Temperature map + wind vectors (sun centered at 180°)
  Bottom-left:  Zonal-mean u-wind (Coriolis signature)
  Bottom-right: Meridional v-wind (Hadley cell)

Static sun → patterns emerge and stabilize cleanly.

Run:  uv run cellsimulations/atmosphere/animate_grid.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import GridGeometry, Params, step_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

N_LAT, N_LON = 36, 72
OMEGA = 0.4
SUN_LON = np.pi      # sun at 180° → centered in plot

FPS = 10
DURATION = 10
TOTAL_FRAMES = FPS * DURATION  # 100
SUBSTEPS = 20

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'output_animations', 'grid')
p = Params(solar=0.15, cooling=0.02, c_sq=0.15, diffuse=0.0, drag=0.02)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    geo = GridGeometry(N_LAT, N_LON)
    T = np.zeros(geo.shape)
    u = np.zeros(geo.shape)
    v = np.zeros(geo.shape)

    os.makedirs(FRAMES_DIR, exist_ok=True)

    lat_deg = np.degrees(geo.lat_grid)
    lon_deg = np.degrees(geo.lon_grid)
    lats = np.linspace(-90, 90, N_LAT)
    skip = (slice(None, None, 2), slice(None, None, 4))

    print(f"Grid animation: {N_LAT}×{N_LON}, ω={OMEGA}, sun at {np.degrees(SUN_LON):.0f}°")
    print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps → {DURATION}s")

    for frame in range(TOTAL_FRAMES):
        for _ in range(SUBSTEPS):
            T, u, v = step_grid(T, u, v, geo, p, SUN_LON, omega=OMEGA)

        # Fresh figure each frame
        fig = plt.figure(figsize=(13, 9), facecolor='#131822')
        gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.3, wspace=0.25)

        # === Top: Temperature + wind vectors (spans both columns) ===
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor('#131822')

        vmax = max(np.abs(T).max(), 0.1)
        ax1.imshow(T, cmap='magma', origin='lower', extent=[0, 360, -90, 90],
                   aspect='auto', vmin=0, vmax=vmax, interpolation='bilinear')

        # Wind vectors
        ws = max(0.02, np.sqrt(u**2 + v**2).max())
        ax1.quiver(lon_deg[skip], lat_deg[skip], u[skip], v[skip],
                   scale=ws * 25, color='white', alpha=0.6, width=0.003,
                   headwidth=4, headlength=3)

        # Sun position indicator
        sun_deg = np.degrees(SUN_LON)
        ax1.axvline(sun_deg, color='#FFD700', lw=2, ls='--', alpha=0.4)
        ax1.text(sun_deg, 88, '☀', fontsize=14, ha='center', va='top', color='#FFD700')

        # Day/night shading (light overlay on night side)
        # Night is where cos(lon - sun) < 0, i.e. |lon - sun| > 90°
        for night_start, night_end in [(0, sun_deg - 90), (sun_deg + 90, 360)]:
            if night_end > night_start:
                ax1.axvspan(night_start, night_end, alpha=0.15, color='#000033')

        ax1.set(xlabel='Longitude (°)', ylabel='Latitude (°)')
        ax1.set_title('Temperature + Wind  (sun at 180°, static)',
                     color='white', fontsize=12)
        ax1.tick_params(colors='#999')
        for spine in ax1.spines.values():
            spine.set_color('#444')

        # === Bottom-left: Zonal-mean u-wind ===
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor('#131822')
        u_zonal = u.mean(axis=1)
        ax2.plot(lats, u_zonal, color='#88ddff', lw=2.5)
        ax2.axhline(0, color='#555', lw=0.8)
        ax2.fill_between(lats, 0, u_zonal, where=(u_zonal > 0),
                        alpha=0.35, color='#ff6b6b', label='Westerly (→ east)')
        ax2.fill_between(lats, 0, u_zonal, where=(u_zonal < 0),
                        alpha=0.35, color='#4ecdc4', label='Easterly (→ west)')
        ax2.set(xlabel='Latitude (°)', ylabel='Zonal wind (u)')
        ax2.set_title('Coriolis: Zonal-Mean Wind', color='white', fontsize=11)
        ax2.set_xlim(-90, 90)
        ylim = max(np.abs(u_zonal).max() * 1.5, 0.05)
        ax2.set_ylim(-ylim, ylim)
        ax2.legend(fontsize=8, loc='upper left', framealpha=0.2,
                  labelcolor='white', facecolor='#1a1a2e')
        ax2.tick_params(colors='#999')
        for spine in ax2.spines.values():
            spine.set_color('#444')
        ax2.grid(True, alpha=0.12)

        # === Bottom-right: Meridional v-wind ===
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor('#131822')
        v_zonal = v.mean(axis=1)
        ax3.plot(lats, v_zonal, color='#ffd700', lw=2.5)
        ax3.axhline(0, color='#555', lw=0.8)
        ax3.fill_between(lats, 0, v_zonal, where=(v_zonal > 0),
                        alpha=0.35, color='#ff6b6b', label='Northward ↑')
        ax3.fill_between(lats, 0, v_zonal, where=(v_zonal < 0),
                        alpha=0.35, color='#4ecdc4', label='Southward ↓')
        ax3.set(xlabel='Latitude (°)', ylabel='Meridional wind (v)')
        ax3.set_title('Hadley Cell: North-South Flow', color='white', fontsize=11)
        ax3.set_xlim(-90, 90)
        ax3.set_ylim(-ylim, ylim)
        ax3.legend(fontsize=8, loc='upper left', framealpha=0.2,
                  labelcolor='white', facecolor='#1a1a2e')
        ax3.tick_params(colors='#999')
        for spine in ax3.spines.values():
            spine.set_color('#444')
        ax3.grid(True, alpha=0.12)

        fig.suptitle(f'Grid Model — Coriolis Organizes Wind Into Latitude Bands  '
                    f'[frame {frame}]',
                    color='white', fontsize=13, fontweight='bold', y=0.98)

        fig.savefig(os.path.join(FRAMES_DIR, f'f_{frame:04d}.png'), dpi=100,
                   facecolor=fig.get_facecolor())
        plt.close(fig)

        if frame % 25 == 0:
            u_mid = u.mean(axis=1)[3 * N_LAT // 4]
            print(f"  {frame}/{TOTAL_FRAMES} ({100*frame//TOTAL_FRAMES}%) "
                  f"| u_45N={u_mid:.3f}")

    print(f"\nFrames → {FRAMES_DIR}/")
    print(f"Stitch: nix shell nixpkgs#ffmpeg --command ffmpeg -framerate {FPS} "
          f"-i {FRAMES_DIR}/f_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p -crf 20 "
          f"{os.path.dirname(FRAMES_DIR)}/grid.mp4")


if __name__ == '__main__':
    main()
