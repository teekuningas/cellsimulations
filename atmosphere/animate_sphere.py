# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Animated 3D sphere — wind patterns from rotating planet physics.

Four panels:
  Top-left:     3D globe surface wind (fixed camera, sun orbits)
  Top-right:    3D globe mid-level wind (same view — compare colors!)
  Bottom-left:  Zonal wind u(lat, alt) heatmap
  Bottom-right: Wind profile u(z) at 30°N showing direction change with altitude

Single rotation parameter drives both Coriolis deflection AND the day/night cycle.
Axial tilt creates asymmetric heating (northern hemisphere summer).

Run:  uv run cellsimulations/atmosphere/animate_sphere.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import Params
from run_sphere import SphereGeometry, step_sphere

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

N_ALT = 8
N_LAT = 24
N_LON = 48

# Single rotation rate — drives both Coriolis AND day/night cycle
ROTATION = 0.25
G = 0.06
TILT = np.radians(0.0)  # equinox — symmetric north/south heating

FPS = 10
DURATION = 20
TOTAL_FRAMES = FPS * DURATION  # 200
SUBSTEPS = 4

FRAMES_DIR = os.path.join(os.path.dirname(__file__), 'output_animations', 'sphere')
p = Params(solar=0.18, cooling=0.02, c_sq=0.15, diffuse=0.0, drag=0.02)


# ═══════════════════════════════════════════════════════════
#  3D Globe Helper
# ═══════════════════════════════════════════════════════════

def make_sphere_mesh(n_lat=40, n_lon=50):
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    return x, y, z, lat_grid, lon_grid


def interpolate_to_sphere(data_2d, sphere_lat, sphere_lon, n_lat, n_lon):
    lat_idx = (sphere_lat + np.pi/2) / np.pi * (n_lat - 1)
    lon_idx = (sphere_lon % (2*np.pi)) / (2*np.pi) * n_lon
    lat_idx = np.clip(lat_idx.astype(int), 0, n_lat - 1)
    lon_idx = np.clip(lon_idx.astype(int), 0, n_lon - 1)
    return data_2d[lat_idx, lon_idx]


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    geo = SphereGeometry(N_ALT, N_LAT, N_LON)
    T = np.zeros(geo.shape)
    u = np.zeros(geo.shape)
    v = np.zeros(geo.shape)
    w = np.zeros(geo.shape)
    sun = 0.0

    os.makedirs(FRAMES_DIR, exist_ok=True)

    sx, sy, sz, slat, slon = make_sphere_mesh(40, 50)
    lats = np.linspace(-90, 90, N_LAT)

    # Sun advances by rotation * dt per physics step
    sun_step = ROTATION * p.dt

    print(f"Sphere animation: {N_ALT}×{N_LAT}×{N_LON}, rotation={ROTATION}, g={G}, tilt={np.degrees(TILT):.1f}°")
    print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps → {DURATION}s")
    print(f"  sun_step={sun_step:.4f} rad/step, {sun_step*SUBSTEPS:.3f} rad/frame")

    # Fixed camera angle
    CAMERA_ELEV = 15
    CAMERA_AZIM = -60

    for frame in range(TOTAL_FRAMES):
        for _ in range(SUBSTEPS):
            T, u, v, w = step_sphere(T, u, v, w, geo, p, sun,
                                     omega=ROTATION, g=G, tilt=TILT)
            sun += sun_step

        sun_angle = sun % (2 * np.pi)

        fig = plt.figure(figsize=(14, 10), facecolor='#131822')

        # Shared color scale for both globes
        u_zonal = u.mean(axis=2)
        vmax_u = max(np.abs(u[0]).max(), np.abs(u[N_ALT//2]).max(), 0.02)

        # === Top-left: 3D globe SURFACE ===
        ax1 = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#131822')
        surf_data = interpolate_to_sphere(u[0], slat, slon, N_LAT, N_LON)
        colors1 = cm.RdBu_r((surf_data + vmax_u) / (2 * vmax_u))
        ax1.plot_surface(sx, sy, sz, facecolors=colors1, alpha=0.95,
                        shade=False, antialiased=False)

        # Sun as gold dot orbiting the globe
        sun_x = 1.8 * np.cos(sun_angle)
        sun_y = 1.8 * np.sin(sun_angle)
        ax1.scatter([sun_x], [sun_y], [0], color='#FFD700', s=200,
                   marker='*', zorder=10)
        # Sun beam line
        ax1.plot([sun_x, np.cos(sun_angle)],
                [sun_y, np.sin(sun_angle)],
                [0, 0], color='#FFD700', alpha=0.3, lw=1)

        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(-1.5, 1.5)
        ax1.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
        ax1.axis('off')
        ax1.set_title('Surface Wind (z=0)',
                     color='white', fontsize=11, pad=-5)

        # === Top-right: 3D globe MID-LEVEL ===
        ax2 = fig.add_subplot(2, 2, 2, projection='3d', facecolor='#131822')
        mid_z = N_ALT // 2
        mid_data = interpolate_to_sphere(u[mid_z], slat, slon, N_LAT, N_LON)
        colors2 = cm.RdBu_r((mid_data + vmax_u) / (2 * vmax_u))
        r_mid = 1.15  # slightly bigger to suggest altitude
        ax2.plot_surface(sx * r_mid, sy * r_mid, sz * r_mid,
                        facecolors=colors2, alpha=0.9,
                        shade=False, antialiased=False)
        # Faint inner planet
        ax2.plot_surface(sx * 0.45, sy * 0.45, sz * 0.45,
                        color='#2a3550', alpha=0.4, shade=False)
        ax2.scatter([sun_x], [sun_y], [0], color='#FFD700', s=200,
                   marker='*', zorder=10)

        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_zlim(-1.5, 1.5)
        ax2.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
        ax2.axis('off')
        ax2.set_title(f'Mid-level Wind (z={mid_z})',
                     color='white', fontsize=11, pad=-5)

        # Color legend between the globes
        fig.text(0.5, 0.55,
                'red = westerly (→east)    white = calm    blue = easterly (→west)',
                color='#aaa', fontsize=9, ha='center')

        # === Bottom-left: u(lat, alt) heatmap ===
        ax3 = fig.add_subplot(2, 2, 3, facecolor='#131822')
        vmax_z = max(np.abs(u_zonal).max(), 0.01)
        im = ax3.imshow(u_zonal, cmap='RdBu_r', aspect='auto',
                       origin='lower', extent=[-90, 90, -0.5, N_ALT-0.5],
                       vmin=-vmax_z, vmax=vmax_z, interpolation='bilinear')
        cb = plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.02)
        cb.ax.tick_params(colors='#999')
        ax3.set_xlabel('Latitude (°)', color='#ccc')
        ax3.set_ylabel('Altitude layer', color='#ccc')
        ax3.set_title('Zonal-Mean Wind u(lat, alt)', color='white', fontsize=11)
        ax3.tick_params(colors='#999')
        ax3.axvline(0, color='white', lw=0.5, ls=':', alpha=0.4)
        for spine in ax3.spines.values():
            spine.set_color('#444')

        # === Bottom-right: Zonal-mean wind profile at 30°N ===
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#131822')
        lat_30N = int(N_LAT * (90 + 30) / 180)
        u_prof = u_zonal[:, lat_30N]

        ax4.plot(u_prof, range(N_ALT), 'o-', color='#FFD700', lw=2.5,
                markersize=7, label='30°N', zorder=5)

        # Shade regions by wind direction
        for z in range(N_ALT):
            color = '#ff6b6b' if u_prof[z] > 0 else '#4ecdc4'
            alpha = 0.12 + 0.08 * abs(u_prof[z]) / max(vmax_z, 0.001)
            ax4.axhspan(z - 0.45, z + 0.45, alpha=min(alpha, 0.3), color=color)

        ax4.axvline(0, color='#666', lw=1, ls='--')
        ax4.set_xlabel('Zonal wind (u)', color='#ccc')
        ax4.set_ylabel('Altitude', color='#ccc')
        ax4.set_title('Wind vs Height (30°N)', color='white', fontsize=11)
        xlim = max(vmax_z * 1.3, 0.03)
        ax4.set_xlim(-xlim, xlim)
        ax4.set_ylim(-0.5, N_ALT - 0.5)

        # Annotate direction
        ax4.text(xlim * 0.6, N_ALT - 1, '→ east', color='#ff6b6b',
                fontsize=9, ha='center', va='center')
        ax4.text(-xlim * 0.6, N_ALT - 1, '← west', color='#4ecdc4',
                fontsize=9, ha='center', va='center')

        ax4.tick_params(colors='#999')
        for spine in ax4.spines.values():
            spine.set_color('#444')
        ax4.grid(True, alpha=0.1, color='#555')

        # Simple title
        fig.suptitle('3D Atmosphere — Equinox (symmetric heating)',
                    color='white', fontsize=12, fontweight='bold', y=0.99)
        fig.text(0.5, 0.01,
                f'frame {frame}  •  sun: {np.degrees(sun_angle):.0f}°',
                color='#555', fontsize=8, ha='center')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(os.path.join(FRAMES_DIR, f'f_{frame:04d}.png'), dpi=100,
                   facecolor=fig.get_facecolor())
        plt.close(fig)

        if frame % 25 == 0:
            print(f"  {frame}/{TOTAL_FRAMES} | "
                  f"u_surf={u_prof[0]:+.3f} u_mid={u_prof[N_ALT//2]:+.3f}")

    print(f"\nFrames → {FRAMES_DIR}/")
    print(f"Stitch: nix shell nixpkgs#ffmpeg --command ffmpeg -framerate {FPS} "
          f"-i {FRAMES_DIR}/f_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p -crf 20 "
          f"{os.path.dirname(FRAMES_DIR)}/sphere.mp4")


if __name__ == '__main__':
    main()
