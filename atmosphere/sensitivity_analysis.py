# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from physics import Params
from run_sphere import SphereGeometry, step_sphere

def run_simulation(N_ALT=8, N_LAT=24, N_LON=48, ROTATION=0.25, G=0.06, TILT=0.0, 
                   solar=0.18, cooling=0.02, c_sq=0.15, drag=0.02, dt=0.4,
                   TOTAL_FRAMES=300, SUBSTEPS=6):
    
    geo = SphereGeometry(N_ALT, N_LAT, N_LON)
    p = Params(solar=solar, cooling=cooling, c_sq=c_sq, diffuse=0.0, drag=drag, dt=dt)
    
    T = np.zeros(geo.shape)
    u = np.zeros(geo.shape)
    v = np.zeros(geo.shape)
    w = np.zeros(geo.shape)
    
    sun = 0.0
    sun_step = ROTATION * p.dt
    
    for frame in range(TOTAL_FRAMES):
        for _ in range(SUBSTEPS):
            T, u, v, w = step_sphere(T, u, v, w, geo, p, sun,
                                     omega=ROTATION, g=G, tilt=TILT)
            sun += sun_step
            
    # Calculate gameplay metrics based on the final frame
    
    # 1. Vertical Reversal Strength
    u_zonal = u.mean(axis=2) # Shape: (n_alt, n_lat)
    
    # Average difference between max and min zonal wind across altitude for each latitude
    reversal_score = np.mean(np.max(u_zonal, axis=0) - np.min(u_zonal, axis=0))
    
    # Calculate sign flips in zonal wind across altitude (average per latitude)
    signs = np.sign(u_zonal)
    sign_flips = np.sum(signs[:-1, :] != signs[1:, :]) / N_LAT
    
    # 2. Patchiness (Longitudinal Variance)
    # Variance of u and v along the longitude axis
    u_variance = np.var(u, axis=2) 
    v_variance = np.var(v, axis=2)
    patchiness_score = np.mean(u_variance + v_variance)
    
    # 3. Overall Wind Speed
    avg_wind_speed = np.mean(np.sqrt(u**2 + v**2 + w**2))
    max_wind_speed = np.max(np.sqrt(u**2 + v**2 + w**2))

    # 4. Energy Stability
    total_E = 0.5 * (np.sum(T**2) + np.sum(u**2) + np.sum(v**2) + np.sum(w**2))

    return {
        'reversal_score': reversal_score,
        'sign_flips': sign_flips,
        'patchiness': patchiness_score,
        'avg_speed': avg_wind_speed,
        'max_speed': max_wind_speed,
        'total_energy': total_E
    }

if __name__ == '__main__':
    print("Running Sensitivity Analysis...")
    
    # Fixing ROTATION mostly, testing a narrow band, varying thermodynamics
    rotations = [0.1, 0.25, 0.4]
    solars = [0.1, 0.18, 0.25]
    coolings = [0.01, 0.02, 0.05]
    drags = [0.01, 0.02, 0.05]
    
    results = []
    
    total_runs = len(rotations) * len(solars) * len(coolings) * len(drags)
    count = 0
    
    print(f"Total configurations to test: {total_runs}")
    print(f"{'ROT':<5} | {'SOL':<5} | {'COL':<5} | {'DRG':<5} | {'REV_SCR':<8} | {'FLIPS':<6} | {'PATCH':<8} | {'AVG_SPD':<8} | {'MAX_SPD':<8} | {'ENERGY':<8}")
    print("-" * 90)
    
    for rot in rotations:
        for sol in solars:
            for cool in coolings:
                for drg in drags:
                    count += 1
                    try:
                        res = run_simulation(ROTATION=rot, solar=sol, cooling=cool, drag=drg)
                        print(f"{rot:<5} | {sol:<5} | {cool:<5} | {drg:<5} | {res['reversal_score']:<8.4f} | {res['sign_flips']:<6.2f} | {res['patchiness']:<8.4f} | {res['avg_speed']:<8.4f} | {res['max_speed']:<8.4f} | {res['total_energy']:<8.1f}")
                        results.append({
                            'rot': rot, 'sol': sol, 'cool': cool, 'drg': drg,
                            **res
                        })
                    except Exception as e:
                        print(f"{rot:<5} | {sol:<5} | {cool:<5} | {drg:<5} | FAILED: {str(e)}")

    print("\nTop 5 Configurations by Vertical Reversal (sign flips + amplitude):")
    results.sort(key=lambda x: (x['sign_flips'], x['reversal_score']), reverse=True)
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"ROT: {r['rot']}, SOL: {r['sol']}, COL: {r['cool']}, DRG: {r['drg']} -> Flips: {r['sign_flips']:.2f}, RevScr: {r['reversal_score']:.4f}, Patch: {r['patchiness']:.4f}, AvgSpd: {r['avg_speed']:.4f}")

    print("\nTop 5 Configurations by Patchiness (Longitudinal Variance):")
    results.sort(key=lambda x: x['patchiness'], reverse=True)
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"ROT: {r['rot']}, SOL: {r['sol']}, COL: {r['cool']}, DRG: {r['drg']} -> Patch: {r['patchiness']:.4f}, Flips: {r['sign_flips']:.2f}, RevScr: {r['reversal_score']:.4f}, AvgSpd: {r['avg_speed']:.4f}")
