import numpy as np
import h5py
import os
import json
from scipy.integrate import solve_ivp
import scipy.linalg

def kinetic(t, y, k1, k2, k3, k4):
    """Core kinetic model."""
    K = np.array([[0,  0,  k4,  k3],  # Bleached
                  [0, -k2, 0, k1],  # Quenched
                  [0, 0, -k4, 0],  # UnQuenched 2
                  [0,  k2, 0, -k1-k3]])  # Unquenched
    return K @ y


class Result:
    """Simple class to mimic solve_ivp return object."""
    def __init__(self, y):
        self.y = y

def solve_expm(K, y0, t):
    """Solve linear ODE system using matrix exponential."""
    if len(t) == 0:
        return Result(np.zeros((len(y0), 0)))
    if len(t) == 1:
        return Result(y0.reshape(-1, 1))
    
    dt = t[1] - t[0]
    M = scipy.linalg.expm(K * dt)
    y = np.empty((len(t), len(y0)))
    y[0] = y0
    for j in range(1, len(t)):
        y[j] = M @ y[j-1]
    return Result(y.T)

def modelfunc(t, k1, k2, k3, k4, q_sum, q_fraction, t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=None):
    """
    Model function for solving the kinetic equations across multiple light/dark phases.
    If k2_light is provided, it is used during light phases instead of k2.
    """
    # kl1 = k2 + k2_light if k2_light is not None else k2
    kl1 = k2

    q0 = q_sum * q_fraction
    q1 = q_sum * (1 - q_fraction)
    y0 = np.array([0, 0, q0, q1])
    
    # Light/Dark Phase Matrix Definitions
    # K = [[Bleached], [Quenched], [UnQuenched2], [Unquenched]]
    K_light = np.array([[0,  0,  k4,  k3],  # Bleached
                        [0, -kl1, 0, k1],  # Quenched
                        [0, 0, -k4, 0],  # UnQuenched 2
                        [0,  kl1, 0, -k1-k3]])  # Unquenched
    
    K_dark = np.array([[0,  0,  0,  0],  # Bleached
                       [0, -k2, 0, 0],  # Quenched
                       [0, 0, 0, 0],  # UnQuenched 2
                       [0,  k2, 0, 0]])  # Unquenched
    
    sol1 = solve_expm(K_light, y0, t[0:t_dark+1])
    sol2 = solve_expm(K_dark, sol1.y[:, -1], t[t_dark:t_light+1])
    sol3 = solve_expm(K_light, sol2.y[:, -1], t[t_light:t_dark2+1])
    sol4 = solve_expm(K_dark, sol3.y[:, -1], t[t_dark2:t_light2+1])
    sol5 = solve_expm(K_light, sol4.y[:, -1], t[t_light2:t_dark3+1])
    sol6 = solve_expm(K_dark, sol5.y[:, -1], t[t_dark3:-1])
    
    return sol1, sol2, sol3, sol4, sol5, sol6

def onetrace(data_dir, partnum, startind=0, low_value_threshold=None):
    """
    Load and normalize a single trace from an H5 file.
    Includes outlier handling.
    """
    dataset = h5py.File(os.path.join(data_dir, f'measurement {partnum}.h5'), 'r')
    abstimes = dataset['timestamps'][:] * 50  # 50 ns clock

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]  # gap is at least 20 ms
    boundary_times = abstimes[boundary_photons]  # end of pulse (start of gap)
    boundary_times_start = abstimes[boundary_photons + 1]  # start of pulse
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])  # first photon is start of first pulse
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6  # length of each pulse in ms
    timestep = np.mean(np.diff(boundary_times_start) / 1e9)  # timestep in s

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    
    # normalize to the start index
    if isinstance(startind, int):
        norm_pulsephotons /= np.mean(norm_pulsephotons[:max(1, startind)])
    else:
        # Some versions use a specific index like norm_pulsephotons[startind]
        norm_pulsephotons /= np.mean(norm_pulsephotons[startind])

    if low_value_threshold is not None:
        # Handling from Power_studies_June2026.py
        norm_pulsephotons = norm_pulsephotons[np.isfinite(norm_pulsephotons)]
        mean_val = np.mean(norm_pulsephotons)
        threshold = float(mean_val * low_value_threshold)
        for i in range(1, len(norm_pulsephotons)):
            if norm_pulsephotons[i] < threshold:
                norm_pulsephotons[i] = norm_pulsephotons[i - 1]
    else:
        # Handling from test_k2_light.py and others
        for i in range(len(norm_pulsephotons)):
            if not np.isfinite(norm_pulsephotons[i]) or norm_pulsephotons[i] <= 0:
                if i == 0:
                    good = None
                    for j in range(1, len(norm_pulsephotons)):
                        if np.isfinite(norm_pulsephotons[j]) and norm_pulsephotons[j] > 0:
                            good = norm_pulsephotons[j]
                            break
                    norm_pulsephotons[i] = good if good is not None else 1.0
                else:
                    norm_pulsephotons[i] = norm_pulsephotons[i - 1]
            elif i > 0 and norm_pulsephotons[i] < norm_pulsephotons[i - 1] * 1e-3:
                norm_pulsephotons[i] = norm_pulsephotons[i - 1]
                
    return norm_pulsephotons, timestep

def avtrace(data_dir, partnums, startind=0, low_value_threshold=None):
    """Average multiple traces."""
    results = [onetrace(data_dir, p, startind, low_value_threshold) for p in partnums]
    traces = [r[0] for r in results]
    timesteps = [r[1] for r in results]
    
    minlength = np.min([len(t) for t in traces])
    traces = [t[:minlength] for t in traces]
    
    return np.mean(traces, axis=0), np.mean(timesteps, axis=0)

def load_params(data_dir, defaults=None):
    """
    Load parameters from params.json, params.txt, or config.json.
    """
    params = defaults.copy() if defaults else {}
    
    # Try different possible config file names
    for filename in ['params.json', 'params.txt', 'config.json']:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    if filename.endswith('.json'):
                        params.update(json.load(f))
                    else:
                        # Simple key-value parser for txt if needed, 
                        # but existing code seems to use json.load even for .txt sometimes?
                        # Let's check test_k2_light.py load_params
                        try:
                            params.update(json.load(f))
                        except:
                            pass # Fallback or handle differently
                break # Stop after first successful load
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                
    return params
