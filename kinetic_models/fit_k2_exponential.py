import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import os
import json
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    'mathtext.fontset': 'stixsans',
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "font.size": 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'legend.fontsize': 7,
})
from scipy.optimize import curve_fit
import kinetic_model

# Select one dataset
# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/29 May 2026/Power study LHCII'
base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/2 June 2026'
# base_data_dir = r'/home/bertus/Documents/Postdoc/Metings/Suurstofprojek/2026/4 June 2026/Thylakoid power study'
# folder_name = '446 mE'
# folder_name = 'LHCII SOD'
folder_name = 'LHCII Control 301 mE'
data_dir = os.path.join(base_data_dir, folder_name)

# Default parameters (from Power_studies_June2026.py)
default_params = {
    'partlist': [0, 1, 2, 3, 4, 5, 6, 7],
    'onlen': 50,
    'offlen': 600,
    'startind': 0,
    'low_value_threshold': 0.1
}

def load_config(folder_path):
    config_file = os.path.join(folder_path, 'config.json')
    params = default_params.copy()
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            params.update(json.load(f))
    return params

params = load_config(data_dir)
partnums = params['partlist']
onlen = params['onlen']
offlen = params['offlen']
startind = params['startind']
low_value_threshold = params['low_value_threshold']

print(f"Loading data from {data_dir}...")
# Use avtrace from kinetic_model (which handles normalization and averaging)
norm_pulsephotons, timestep = kinetic_model.avtrace(data_dir, partnums, startind=startind, low_value_threshold=low_value_threshold)

# In Power_studies_June2026.py, startind is used to slice norm_pulsephotons
norm_pulsephotons = norm_pulsephotons[startind:]

# Identify the first dark phase
# The light phase is onlen. Dark phase starts at t_dark = onlen.
# Length of dark phase is offlen.
t_dark_start = onlen
t_dark_end = onlen + offlen
    
# Check if we have enough data
if t_dark_end > len(norm_pulsephotons):
    print(f"Warning: Trace is shorter ({len(norm_pulsephotons)}) than expected end of dark phase ({t_dark_end}).")
    t_dark_end = len(norm_pulsephotons)

# Extract the dark part
dark_part = norm_pulsephotons[t_dark_start:t_dark_end]
t_dark = np.arange(len(dark_part)) * timestep

# Bi-exponential fit function: f(t) = A1 * exp(-k1 * t) + A2 * exp(-k2 * t) + C
def bi_exp_fit(t, A1, k1, A2, k2, C):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C

# Initial guesses
# Assume two components: one faster, one slower
C_guess = dark_part[-1]
total_amp = dark_part[0] - C_guess
A1_guess = total_amp * 0.5
A2_guess = total_amp * 0.5
k1_guess = 1.0   # Fast component
k2_guess = 0.1   # Slow component

p0 = [A1_guess, k1_guess, A2_guess, k2_guess, C_guess]

print("Fitting bi-exponential to dark phase...")
try:
    # Adding bounds to ensure rates are positive and meaningful
    bounds = ([-np.inf, 0, -np.inf, 0, -np.inf], [0, 100, 0, 100, np.inf])
    popt, pcov = curve_fit(bi_exp_fit, t_dark, dark_part, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    
    A1, k1, A2, k2, C = popt
    A1_err, k1_err, A2_err, k2_err, C_err = perr
    
    # Sort by rates to consistently label k1 as fast and k2 as slow (or vice-versa)
    # Actually, let's just print them as they come.
    
    print("\nFit Results (Bi-exponential):")
    print(f"Component 1: k1 = {k1:.4f} ± {k1_err:.4f} s⁻¹, Tau1 = {1/k1:.4f} s, A1 = {A1:.4f}")
    print(f"Component 2: k2 = {k2:.4f} ± {k2_err:.4f} s⁻¹, Tau2 = {1/k2:.4f} s, A2 = {A2:.4f}")
    print(f"Offset (C) = {C:.4f} ± {C_err:.4f}")

    # Average lifetime and k calculations
    # Standard definitions for bi-exponential decay: f(t) = sum A_i exp(-t/tau_i)
    tau1 = 1/k1
    tau2 = 1/k2
    
    # 1. Amplitude-weighted average lifetime: <tau>_amp = (A1*tau1 + A2*tau2) / (A1 + A2)
    tau_amp_weighted = (A1 * tau1 + A2 * tau2) / (A1 + A2)
    
    # 2. Intensity-weighted (Area-weighted) average lifetime: <tau>_int = (A1*tau1^2 + A2*tau2^2) / (A1*tau1 + A2*tau2)
    # This is often what is meant by "average lifetime" in fluorescence spectroscopy
    tau_int_weighted = (A1 * tau1**2 + A2 * tau2**2) / (A1 * tau1 + A2 * tau2)

    # Error propagation using the Delta Method: sigma_f^2 = J * V * J^T
    # where J is the Jacobian (gradient) of the function f with respect to parameters
    # and V is the covariance matrix.
    
    # Params: [A1, k1, A2, k2, C]
    
    def get_tau_amp(p):
        a1, k1_, a2, k2_, c = p
        return (a1/k1_ + a2/k2_) / (a1 + a2)
        
    def get_tau_int(p):
        a1, k1_, a2, k2_, c = p
        return (a1/k1_**2 + a2/k2_**2) / (a1/k1_ + a2/k2_)

    def get_k_amp(p):
        a1, k1_, a2, k2_, c = p
        return (a1*k1_ + a2*k2_) / (a1 + a2)

    def get_k_int(p):
        a1, k1_, a2, k2_, c = p
        return (a1 + a2) / (a1/k1_ + a2/k2_)

    def get_gradient(func, p, eps=1e-6):
        grad = np.zeros_like(p)
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += eps
            p_minus = p.copy()
            p_minus[i] -= eps
            grad[i] = (func(p_plus) - func(p_minus)) / (2 * eps)
        return grad

    grad_amp = get_gradient(get_tau_amp, popt)
    tau_amp_err = np.sqrt(grad_amp @ pcov @ grad_amp)
    
    grad_int = get_gradient(get_tau_int, popt)
    tau_int_err = np.sqrt(grad_int @ pcov @ grad_int)

    grad_k_amp = get_gradient(get_k_amp, popt)
    k_amp_err = np.sqrt(grad_k_amp @ pcov @ grad_k_amp)

    grad_k_int = get_gradient(get_k_int, popt)
    k_int_err = np.sqrt(grad_k_int @ pcov @ grad_k_int)

    print(f"\nAverage Lifetimes:")
    print(f"Amplitude-weighted <tau> = {tau_amp_weighted:.4f} ± {tau_amp_err:.4f} s")
    print(f"Intensity-weighted <tau> = {tau_int_weighted:.4f} ± {tau_int_err:.4f} s")

    # Keep original k averages for reference, but update to show they are rates
    k_amp_weighted = (A1 * k1 + A2 * k2) / (A1 + A2)
    k_int_weighted = (A1 + A2) / (A1 / k1 + A2 / k2)
    
    print(f"\nAverage Rates:")
    print(f"Amplitude-weighted average k = {k_amp_weighted:.4f} ± {k_amp_err:.4f} s⁻¹")
    print(f"Intensity-weighted average k = {k_int_weighted:.4f} ± {k_int_err:.4f} s⁻¹")

    # Residuals
    residuals = dark_part - bi_exp_fit(t_dark, *popt)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot
    ax1.plot(t_dark, dark_part, 'ko', label='Experimental Data (Dark Phase)', markersize=4)
    ax1.plot(t_dark, bi_exp_fit(t_dark, *popt), 'r-', label='Bi-exponential Fit', linewidth=2)
    # ax1.plot(t_dark, A1 * np.exp(-k1 * t_dark) + C, 'b--', label=f'Fast component (k1={k1:.3f})', alpha=0.5)
    # ax1.plot(t_dark, A2 * np.exp(-k2 * t_dark) + C, 'g--', label=f'Slow component (k2={k2:.3f})', alpha=0.5)
    ax1.set_ylabel('Normalized Photon Count')
    ax1.set_title(f'First Dark Phase Bi-exponential Fit - {folder_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2.plot(t_dark, residuals, 'bo', markersize=4, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_plot = f"k2_bifit_residuals_{folder_name.replace(' ', '_')}.png"
    plt.savefig(output_plot)
    print(f"\nPlot with residuals saved to {output_plot}")
    plt.show()

except Exception as e:
    print(f"Error during fitting: {e}")