import sys
import os

# Add the project root to sys.path to allow imports from there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
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


def kinetic_2q(t, y, k1, kr1, kr2, k3, k4, f):
    """
    States:
    0 : Bleached
    1 : Q1
    2 : Q2
    3 : Unquenched2
    4 : Unquenched
    """

    K = np.array([
        [0,        0,      0,      k4,        k3],
        [0,    -kr1,      0,       0,    f*k1],
        [0,        0,  -kr2,       0, (1-f)*k1],
        [0,        0,      0,     -k4,       0],
        [0,     kr1,    kr2,       0,    -k1-k3]
    ])

    return K @ y


class Result:
    """Simple class to mimic solve_ivp return object."""
    def __init__(self, y):
        self.y = y

def solve_expm(M, y0, n):
    """Solve linear ODE system using matrix exponential properties."""
    # if n <= 0:
    #     return Result(np.zeros((len(y0), 1)) if len(y0) > 0 else np.zeros((0, 1)))
    
    # Ensure y0 has the correct shape for broadcasting if n=1
    # but the code below handles it via np.arange(n)
    
    # if n == 1:
    #     return Result(y0.reshape(-1, 1))
    
    # y(j) = M^j * y0
    # To avoid repeated matrix multiplications, we use the property:
    # y(j) = V * D^j * V^-1 * y0
    try:
        vals, vecs = scipy.linalg.eig(M)
        ivecs = scipy.linalg.inv(vecs)
        c = ivecs.dot(y0)
        
        # vals is (4,), vecs is (4,4), c is (4,)
        # we want y[j] = vecs @ (vals**j * c)
        j_range = np.arange(n)
        # powers shape: (n, 4)
        powers = vals[np.newaxis, :] ** j_range[:, np.newaxis]
        # res shape: (n, 4)
        res = (powers * c[np.newaxis, :]) @ vecs.T
        return Result(res.real.T)
    except Exception:
        # Fallback to iterative if diagonalization fails
        y = np.empty((n, len(y0)))
        y[0] = y0
        y_prev = y0
        for j in range(1, n):
            y_curr = M.dot(y_prev)
            y[j] = y_curr
            y_prev = y_curr
        return Result(y.T)

def modelfunc(t, k1, k2, k3, k4, q_sum, q_fraction, t_dark, t_light, t_dark2, t_light2, t_dark3, k2_light=None,
              debug=False):
    """
    Model function for solving the kinetic equations across multiple light/dark phases.
    If k2_light is provided, it is used during light phases instead of k2.
    """
    kl1 = k2 + k2_light if k2_light is not None else k2

    q0 = q_sum * q_fraction
    q1 = q_sum * (1 - q_fraction)
    y0 = np.array([0, 0, q0, q1])
    
    # Light/Dark Phase Matrix Definitions
    K_light = np.array([[0,  0,  k4,  k3],  # Bleached
                        [0, -kl1, 0, k1],  # Quenched
                        [0, 0, -k4, 0],  # UnQuenched 2
                        [0,  kl1, 0, -k1-k3]])  # Unquenched
    
    K_dark = np.array([[0,  0,  0,  0],  # Bleached
                       [0, -k2, 0, 0],  # Quenched
                       [0, 0, 0, 0],  # UnQuenched 2
                       [0,  k2, 0, 0]])  # Unquenched
    
    dt = t[1] - t[0]
    M_light = scipy.linalg.expm(K_light * dt)
    M_dark = scipy.linalg.expm(K_dark * dt)

    sol1 = solve_expm(M_light, y0, t_dark + 1)
    sol2 = solve_expm(M_dark, sol1.y[:, -1], t_light - t_dark + 1)
    sol3 = solve_expm(M_light, sol2.y[:, -1], t_dark2 - t_light + 1)
    if debug:
        print(t_dark2, t_light)
        print(sol1.y)
        print(sol2.y)
        print(sol3.y)
    sol4 = solve_expm(M_dark, sol3.y[:, -1], t_light2 - t_dark2 + 1)
    sol5 = solve_expm(M_light, sol4.y[:, -1], t_dark3 - t_light2 + 1)
    sol6 = solve_expm(M_dark, sol5.y[:, -1], len(t) - 1 - t_dark3)
    
    return sol1, sol2, sol3, sol4, sol5, sol6

def modelfunc_2q(t, k1, kr1, kr2, k3, k4, f, q_sum, q_fraction, t_dark, t_light, t_dark2, t_light2, t_dark3,
                 debug=False):
    """
    Model function for the 2-quenched-state kinetic model across multiple phases.
    """
    q0 = q_sum * q_fraction
    q1 = q_sum * (1 - q_fraction)
    # y0: [Bleached, Q1, Q2, Unquenched2, Unquenched]
    y0 = np.array([0, 0, 0, q0, q1])

    K_light = np.array([
        [0,    0,    0,    k4,    k3],
        [0, -kr1,    0,     0,  f*k1],
        [0,    0, -kr2,     0, (1-f)*k1],
        [0,    0,    0,   -k4,     0],
        [0,  kr1,  kr2,     0, -k1-k3]
    ])

    K_dark = np.array([
        [0,    0,    0,    0,    0],
        [0, -kr1,    0,     0,    0],
        [0,    0, -kr2,     0,    0],
        [0,    0,    0,    0,    0],
        [0,  kr1,  kr2,     0,    0]
    ])

    dt = t[1] - t[0]
    M_light = scipy.linalg.expm(K_light * dt)
    M_dark = scipy.linalg.expm(K_dark * dt)

    sol1 = solve_expm(M_light, y0, t_dark + 1)
    sol2 = solve_expm(M_dark, sol1.y[:, -1], t_light - t_dark + 1)
    sol3 = solve_expm(M_light, sol2.y[:, -1], t_dark2 - t_light + 1)
    sol4 = solve_expm(M_dark, sol3.y[:, -1], t_light2 - t_dark2 + 1)
    sol5 = solve_expm(M_light, sol4.y[:, -1], t_dark3 - t_light2 + 1)
    sol6 = solve_expm(M_dark, sol5.y[:, -1], len(t) - 1 - t_dark3)

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

def avtrace(data_dir, partnums, startind=0, low_value_threshold=None, return_all=False):
    """Average multiple traces. If return_all is True, also returns the individual traces."""
    results = [onetrace(data_dir, p, startind, low_value_threshold) for p in partnums]
    traces = [r[0] for r in results]
    timesteps = [r[1] for r in results]
    
    minlength = np.min([len(t) for t in traces])
    traces = [t[:minlength] for t in traces]
    
    avg_trace = np.mean(traces, axis=0)
    avg_timestep = np.mean(timesteps, axis=0)
    
    if return_all:
        return avg_trace, avg_timestep, traces
    return avg_trace, avg_timestep

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
