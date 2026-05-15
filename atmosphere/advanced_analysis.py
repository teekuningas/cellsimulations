# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import Params
from run_sphere import SphereGeometry, _horiz_gradient, _vert_gradient, _advect_2d, _advect_vertical

def step_sphere_advanced(T, u, v, w, geo, p, sun_lon, omega=0.4, g=0.06, tilt=0.0, surface_drag_only=False):
    """Modified step_sphere to allow surface-only drag."""
    T = T + p.dt * p.solar * geo.solar(sun_lon, tilt=tilt)
    T = T * (1 - p.dt * p.cooling)

    for z in range(geo.n_alt):
        dTdx, dTdy = _horiz_gradient(T[z])
        u[z] = u[z] - p.dt * p.c_sq * dTdx
        v[z] = v[z] - p.dt * p.c_sq * dTdy

    dTdz = _vert_gradient(T)
    w = w - p.dt * p.c_sq * dTdz

    f = 2 * omega * geo.sin_lat * p.dt
    c, s = np.cos(f), np.sin(f)
    for z in range(geo.n_alt):
        u_old, v_old = u[z].copy(), v[z].copy()
        u[z] = u_old * c + v_old * s
        v[z] = -u_old * s + v_old * c

    w = w - p.dt * g

    u0, v0, w0 = u.copy(), v.copy(), w.copy()
    for z in range(geo.n_alt):
        T[z] = _advect_2d(T[z], u0[z], v0[z], p.dt)
        u[z] = _advect_2d(u[z], u0[z], v0[z], p.dt)
        v[z] = _advect_2d(v[z], u0[z], v0[z], p.dt)
        w[z] = _advect_2d(w[z], u0[z], v0[z], p.dt)

    T = _advect_vertical(T, w0, p.dt)
    u = _advect_vertical(u, w0, p.dt)
    v = _advect_vertical(v, w0, p.dt)
    w = _advect_vertical(w, w0, p.dt)

    # ADVANCED FRICTION
    if surface_drag_only:
        # Drag only at z=0
        u[0] = u[0] * (1 - p.dt * p.drag)
        v[0] = v[0] * (1 - p.dt * p.drag)
        w[0] = w[0] * (1 - p.dt * p.drag)
    else:
        # Uniform drag everywhere
        u = u * (1 - p.dt * p.drag)
        v = v * (1 - p.dt * p.drag)
        w = w * (1 - p.dt * p.drag)

    w[0, :, :] = 0
    w[-1, :, :] = 0
    v[:, 0, :] = 0
    v[:, -1, :] = 0

    return T, u, v, w

def run_advanced_sim(ROTATION=0.25, TILT_DEG=0.0, drag=0.02, surface_drag_only=False, 
                     solar=0.25, cooling=0.02):
    
    geo = SphereGeometry(8, 24, 48)
    p = Params(solar=solar, cooling=cooling, c_sq=0.15, diffuse=0.0, drag=drag, dt=0.4)
    
    T = np.zeros(geo.shape)
    u = np.zeros(geo.shape)
    v = np.zeros(geo.shape)
    w = np.zeros(geo.shape)
    
    sun = 0.0
    sun_step = ROTATION * p.dt
    tilt_rad = np.radians(TILT_DEG)
    
    for frame in range(300):
        for _ in range(6):
            T, u, v, w = step_sphere_advanced(T, u, v, w, geo, p, sun,
                                              omega=ROTATION, g=0.06, tilt=tilt_rad,
                                              surface_drag_only=surface_drag_only)
            sun += sun_step
            
    # Metrics
    # Upper-level wind speed (z=6)
    upper_wind = np.mean(np.sqrt(u[-2]**2 + v[-2]**2))
    # Surface wind speed (z=0)
    surface_wind = np.mean(np.sqrt(u[0]**2 + v[0]**2))
    # Hemisphere asymmetry: mean T in Northern vs Southern Hemisphere
    NH_T = np.mean(T[:, 12:, :])
    SH_T = np.mean(T[:, :12, :])
    asymmetry = abs(NH_T - SH_T)
    # Cross-equatorial flow (v wind at equator)
    cross_eq_v = np.mean(np.abs(v[:, 12, :]))
    
    return {
        'upper_wind': upper_wind,
        'surface_wind': surface_wind,
        'asymmetry': asymmetry,
        'cross_eq_v': cross_eq_v
    }

if __name__ == '__main__':
    print("Running Advanced Sensitivity Analysis...")
    print(f"{'TILT(°)':<8} | {'DRG_TYPE':<10} | {'UPPER_SPD':<10} | {'SURF_SPD':<10} | {'ASYM(NH-SH)':<11} | {'CROSS_EQ_V':<10}")
    print("-" * 75)
    
    tilts = [0.0, 23.5, 45.0]
    drag_types = [(0.02, False, "Uniform"), (0.05, True, "Surface-Only")]
    
    for tilt in tilts:
        for drag_val, is_surf, name in drag_types:
            res = run_advanced_sim(TILT_DEG=tilt, drag=drag_val, surface_drag_only=is_surf)
            print(f"{tilt:<8.1f} | {name:<10} | {res['upper_wind']:<10.4f} | {res['surface_wind']:<10.4f} | {res['asymmetry']:<11.4f} | {res['cross_eq_v']:<10.4f}")
