import sys
import os

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add the project root to sys.path to allow imports of utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.stats as stats

FIT_WITH_INTERCEPT = True
FIT_K2_LINEAR = False  # Set to True to perform linear fit on k2 (if light-dependent), False for horizontal line / constant k2
PLOT_POWER_DEPENDENT_K2 = True  # Set to True to plot power-dependent average k2 data points, False for horizontal line

import utils

utils.setup_plotting()

def get_numeric(series):
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or series.dtype in ['O', 'object', 'string', object]:
        return series.astype(str).str.split(' ±').str[0].astype(float)
    return series.astype(float)

def get_error(series):
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or series.dtype in ['O', 'object', 'string', object]:
        # Extract the error part, handle cases where no error is present
        parts = series.astype(str).str.split(' ±')
        return parts.apply(lambda x: float(x[1]) if len(x) > 1 else 0.0)
    return pd.Series(0.0, index=series.index)

def linear_fit(x, y, weights=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    reg_res = stats.linregress(x, y)
    print(reg_res.pvalue)
    if n > 1:
        r, _ = stats.pearsonr(x, y)
    else:
        r = np.nan
    if FIT_WITH_INTERCEPT:
        p, cov = np.polyfit(x, y, 1, w=weights, cov=True)
        m, b = p[0], p[1]
        k_params = 2
    else:
        # Manual calculation for zero-intercept fit: y = m * x
        x_sq_sum = np.sum(x**2)
        m = np.sum(x * y) / x_sq_sum if x_sq_sum != 0 else 0.0
        b = 0.0
        # Residual variance estimate
        resid = y - m * x
        sigma2 = np.sum(resid**2) / (n - 1) if n > 1 else 0
        m_var = sigma2 / x_sq_sum if x_sq_sum != 0 else 0
        cov = np.array([[m_var, 0], [0, 0]])
        k_params = 1

    residuals = y - (m * x + b)
    rss = np.sum(residuals**2)
    bic = n * np.log(rss / n) + k_params * np.log(n) if (rss > 0 and n > 0) else np.nan
    return m, b, cov, r, rss, bic, n

def get_fit_bounds(x_range, m, b, cov):
    y_fit = m * x_range + b
    # Variance propagation for y = mx + b
    # var(y) = x^2 * var(m) + var(b) + 2 * x * cov(m, b)
    var_y = (x_range**2 * cov[0,0]) + (2 * x_range * cov[0,1]) + cov[1,1]
    sig_y = np.sqrt(np.maximum(var_y, 0))
    return y_fit - sig_y, y_fit + sig_y

def print_bic_summary(fits_dict, title="Linear Fits & BIC Summary"):
    """
    Prints a detailed table and total BIC for all linear fits in fits_dict.
    fits_dict format: { 'Rate': {'N': n, 'k': k, 'm': m, 'b': b, 'r': r, 'rss': rss, 'bic': bic} }
    """
    if not fits_dict:
        return

    records = []
    total_n = 0
    total_k = 0
    total_rss = 0.0
    bic_sum = 0.0

    for name, info in fits_dict.items():
        n = info['N']
        k = info['k']
        m = info['m']
        b = info['b']
        r = info['r']
        rss = info['rss']
        bic = info['bic']

        total_n += n
        total_k += k
        total_rss += rss
        bic_sum += bic

        records.append({
            'Rate': name,
            'N': n,
            'k': k,
            'Slope (m)': f"{m:.4e}",
            'Intercept (b)': f"{b:.4e}",
            'r': f"{r:.4f}" if np.isfinite(r) else "N/A",
            'r²': f"{r**2:.4f}" if np.isfinite(r) else "N/A",
            'RSS': f"{rss:.6e}",
            'BIC': f"{bic:.2f}"
        })

    bic_joint = total_n * np.log(total_rss / total_n) + total_k * np.log(total_n) if (total_rss > 0 and total_n > 0) else np.nan

    df_summary = pd.DataFrame(records)
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print(f"{'-'*70}")
    print(f" Total Data Points (N):  {total_n}")
    print(f" Total Parameters (K):   {total_k}")
    print(f" Total RSS:              {total_rss:.6e}")
    print(f" Joint / Combined BIC:   {bic_joint:.2f}  [N*ln(RSS/N) + K*ln(N)]")
    print(f" Sum of Individual BICs: {bic_sum:.2f}  [Σ BIC_i]")
    print(f"{'='*70}\n")

def get_k2_average(df):
    """
    Computes average k2 and its uncertainty for a dataframe.
    Handles 2Q model (Kr1, Kr2, f, with optional Kr1_light, Kr2_light),
    1Q model (K2, with optional K2_light), or single Kr1.
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Check for 2Q model with Kr1 and Kr2
    if 'Kr1 (s⁻¹)' in df.columns and 'Kr2 (s⁻¹)' in df.columns:
        kr1 = get_numeric(df['Kr1 (s⁻¹)'])
        kr1_err = get_error(df['Kr1 (s⁻¹)'])
        kr2 = get_numeric(df['Kr2 (s⁻¹)'])
        kr2_err = get_error(df['Kr2 (s⁻¹)'])
        f_val = get_numeric(df['f']) if 'f' in df.columns else pd.Series(0.5, index=df.index)
        f_err = get_error(df['f']) if 'f' in df.columns else pd.Series(0.0, index=df.index)

        if 'Kr1_light (s⁻¹)' in df.columns and 'Kr2_light (s⁻¹)' in df.columns:
            kr1_l = get_numeric(df['Kr1_light (s⁻¹)'])
            kr1_l_err = get_error(df['Kr1_light (s⁻¹)'])
            kr2_l = get_numeric(df['Kr2_light (s⁻¹)'])
            kr2_l_err = get_error(df['Kr2_light (s⁻¹)'])
            kr1_tot = kr1 + kr1_l
            kr2_tot = kr2 + kr2_l
            kr1_tot_err = np.sqrt(kr1_err**2 + kr1_l_err**2)
            kr2_tot_err = np.sqrt(kr2_err**2 + kr2_l_err**2)
        else:
            kr1_tot = kr1
            kr2_tot = kr2
            kr1_tot_err = kr1_err
            kr2_tot_err = kr2_err

        tau1 = 1.0 / np.where(kr1_tot != 0, kr1_tot, np.nan)
        tau2 = 1.0 / np.where(kr2_tot != 0, kr2_tot, np.nan)
        tau_av = f_val * tau1 + (1.0 - f_val) * tau2
        k2_av = 1.0 / np.where(tau_av != 0, tau_av, np.nan)

        # Error propagation via delta method
        tau1_err = kr1_tot_err / (kr1_tot**2)
        tau2_err = kr2_tot_err / (kr2_tot**2)
        tau_av_err = np.sqrt((f_val * tau1_err)**2 + ((1.0 - f_val) * tau2_err)**2 + ((tau1 - tau2) * f_err)**2)
        k2_av_err = tau_av_err * (k2_av**2)
        return pd.Series(k2_av, index=df.index), pd.Series(k2_av_err, index=df.index)

    # Check for 1Q model with K2
    elif 'K2 (s⁻¹)' in df.columns:
        k2 = get_numeric(df['K2 (s⁻¹)'])
        k2_err = get_error(df['K2 (s⁻¹)'])
        if 'K2_light (s⁻¹)' in df.columns:
            k2_l = get_numeric(df['K2_light (s⁻¹)'])
            k2_l_err = get_error(df['K2_light (s⁻¹)'])
            k2_tot = k2 + k2_l
            k2_tot_err = np.sqrt(k2_err**2 + k2_l_err**2)
        else:
            k2_tot = k2
            k2_tot_err = k2_err
        return pd.Series(k2_tot, index=df.index), pd.Series(k2_tot_err, index=df.index)

    # Fallback if only Kr1 is present (e.g. with AA)
    elif 'Kr1 (s⁻¹)' in df.columns:
        kr1 = get_numeric(df['Kr1 (s⁻¹)'])
        kr1_err = get_error(df['Kr1 (s⁻¹)'])
        if 'Kr1_light (s⁻¹)' in df.columns:
            kr1_l = get_numeric(df['Kr1_light (s⁻¹)'])
            kr1_l_err = get_error(df['Kr1_light (s⁻¹)'])
            k2_tot = kr1 + kr1_l
            k2_tot_err = np.sqrt(kr1_err**2 + kr1_l_err**2)
        elif 'K2_light (s⁻¹)' in df.columns:
            k2_l = get_numeric(df['K2_light (s⁻¹)'])
            k2_l_err = get_error(df['K2_light (s⁻¹)'])
            k2_tot = kr1 + k2_l
            k2_tot_err = np.sqrt(kr1_err**2 + k2_l_err**2)
        else:
            k2_tot = kr1
            k2_tot_err = kr1_err
        return pd.Series(k2_tot, index=df.index), pd.Series(k2_tot_err, index=df.index)

    return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)


results_dir = os.path.join(project_root, 'results')
df_no_aa = pd.DataFrame()
df_with_aa = pd.DataFrame()
df_thylakoid_no_aa = pd.DataFrame()

try:
    path = os.path.join(results_dir, 'data_no_aa_lhcii_2q.pkl')
    with open(path, 'rb') as f:
        df_no_aa = pickle.load(f)
except FileNotFoundError:
    df_no_aa = pd.DataFrame()
    print("data_no_aa_lhcii.pkl not found in results/")

try:
    path = os.path.join(results_dir, 'data_with_aa_lhcii.pkl')
    with open(path, 'rb') as f:
        df_with_aa = pickle.load(f)
except FileNotFoundError:
    df_with_aa = pd.DataFrame()
    print("data_with_aa_lhcii.pkl not found in results/")

try:
    path = os.path.join(results_dir, '../results/data_no_aa.pkl')
    with open(path, 'rb') as f:
        df_thylakoid_no_aa = pickle.load(f)
except FileNotFoundError:
    print("data_no_aa.pkl not found in results/")

# Base data directory for saving plots (can be customized or loaded from data if available)
# In the original script, it was base_data_dir. We'll use current dir or a default.
base_data_dir = '.' 

# Create figure for "with AA" data: single axis for k1, k3, k4
# if not df_with_aa.empty:
#     fig_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 70 / 25.4))
#     fits_dict_aa = {}
#
#     x_aa = df_with_aa['Power (mE)'].values
#     x_extrap = np.linspace(min(2, x_aa.min()), max(2, x_aa.max()), 100)
#
#     # Plot K3 (with AA)
#     k3_aa = get_numeric(df_with_aa['K3 (s⁻¹)'])
#     k3_plot = ax1.errorbar(x_aa, k3_aa,
#                 yerr=get_error(df_with_aa['K3 (s⁻¹)']), fmt='^', label=r'$k_3$',
#                 linewidth=1, markersize=4, alpha=1, color='C1', capsize=3)
#     # Linear fit for K3
#     m3, b3, cov3, r3, rss3, bic3, n3 = linear_fit(x_aa, k3_aa)
#     fits_dict_aa['k3'] = {'N': n3, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m3, 'b': b3, 'r': r3, 'rss': rss3, 'bic': bic3}
#     k3_extrap = m3 * 2 + b3
#     y_low3, y_high3 = get_fit_bounds(x_extrap, m3, b3, cov3)
#     ax1.plot(x_extrap, m3 * x_extrap + b3, 'C1--', label=None)
#     ax1.fill_between(x_extrap, y_low3, y_high3, color='C1', alpha=0.2)
#
#     # Plot K4 (with AA)
#     k4_aa = get_numeric(df_with_aa['K4 (s⁻¹)'])
#     k4_plot = ax1.errorbar(x_aa,  k4_aa,
#                 yerr=get_error(df_with_aa['K4 (s⁻¹)']), fmt='v', label=r'$k_4$',
#                 linewidth=1, markersize=4, alpha=1, color='C2', capsize=3)
#     # Linear fit for K4
#     m4, b4, cov4, r4, rss4, bic4, n4 = linear_fit(x_aa, k4_aa)
#     fits_dict_aa['k4'] = {'N': n4, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m4, 'b': b4, 'r': r4, 'rss': rss4, 'bic': bic4}
#     k4_extrap = m4 * 2 + b4
#     y_low4, y_high4 = get_fit_bounds(x_extrap, m4, b4, cov4)
#     ax1.plot(x_extrap, m4 * x_extrap + b4, 'C2--', label=None)
#     ax1.fill_between(x_extrap, y_low4, y_high4, color='C2', alpha=0.2)
#
#     # Plot K1 (with AA)
#     k1_aa = get_numeric(df_with_aa['K1 (s⁻¹)'])
#     k1_plot = ax1.errorbar(x_aa, k1_aa,
#                  yerr=get_error(df_with_aa['K1 (s⁻¹)']), fmt='o', label=r'$k_1$',
#                  linewidth=1, markersize=4, alpha=1, color='C0', capsize=3)
#     # Linear fit for K1
#     m1, b1, cov1, r1, rss1, bic1, n1 = linear_fit(x_aa, k1_aa)
#     fits_dict_aa['k1'] = {'N': n1, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m1, 'b': b1, 'r': r1, 'rss': rss1, 'bic': bic1}
#     k1_extrap = m1 * 2 + b1
#     y_low1, y_high1 = get_fit_bounds(x_extrap, m1, b1, cov1)
#     ax1.plot(x_extrap, m1 * x_extrap + b1, 'C0--', label=None)
#     ax1.fill_between(x_extrap, y_low1, y_high1, color='C0', alpha=0.2)
#
#     k2_aa, k2_aa_err = get_k2_average(df_with_aa)
#     if FIT_K2_LINEAR:
#         k2_plot = ax1.errorbar(x_aa, k2_aa,
#                                yerr=k2_aa_err, fmt='s', label=r'$k_2$',
#                                linewidth=1, markersize=4, alpha=1, color='C3', capsize=3)
#         m2_aa, b2_aa, cov2_aa, r2_aa, rss2_aa, bic2_aa, n2_aa = linear_fit(x_aa, k2_aa)
#         fits_dict_aa['k2'] = {'N': n2_aa, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m2_aa, 'b': b2_aa, 'r': r2_aa, 'rss': rss2_aa, 'bic': bic2_aa}
#         k2_extrap = m2_aa * 2 + b2_aa
#         y_low2, y_high2 = get_fit_bounds(x_extrap, m2_aa, b2_aa, cov2_aa)
#         ax1.plot(x_extrap, m2_aa * x_extrap + b2_aa, 'C3--', label=None)
#         ax1.fill_between(x_extrap, y_low2, y_high2, color='C3', alpha=0.2)
#     elif PLOT_POWER_DEPENDENT_K2:
#         k2_plot = ax1.errorbar(x_aa, k2_aa,
#                                yerr=k2_aa_err, fmt='s', label=r'$k_2$',
#                                linewidth=1, markersize=4, alpha=1, color='C3', capsize=3)
#         k2_val = k2_aa.mean() if not k2_aa.empty else 0.0
#         ax1.axhline(k2_val, color='C3', linestyle='--', label=r'$k_2$')
#     else:
#         if 'Kr1 (s⁻¹)' in df_with_aa.columns and 'Kr2 (s⁻¹)' in df_with_aa.columns:
#             kr1_aa_val = get_numeric(df_with_aa['Kr1 (s⁻¹)']).mean()
#             kr2_aa_val = get_numeric(df_with_aa['Kr2 (s⁻¹)']).mean()
#             k2_plot = ax1.axhline(kr1_aa_val, color='C3', linestyle='--', label=r'$k_2$')
#             ax1.axhline(kr2_aa_val, color='C3', linestyle='--')
#         else:
#             k2_aa_val = k2_aa.iloc[0] if not k2_aa.empty else 0.0
#             k2_plot = ax1.axhline(k2_aa_val, color='C3', linestyle='--', label=r'$k_2$')
#
#     plot_overlay = False
#     # Overlay Non-AA data on the AA plot
#     if not df_no_aa.empty and plot_overlay:
#         x_no_aa_overlay = df_no_aa['Power (mE)'].values
#
#         # K3 (no AA) overlay
#         k3_no_aa_overlay = get_numeric(df_no_aa['K3 (s⁻¹)'])
#         ax1.errorbar(x_no_aa_overlay, k3_no_aa_overlay,
#                     yerr=get_error(df_no_aa['K3 (s⁻¹)']), fmt='^',
#                     linewidth=1, markersize=4, alpha=0.3, color='C1', capsize=3, label=None)
#         m3_no_aa, b3_no_aa, cov3_no_aa, *_ = linear_fit(x_no_aa_overlay, k3_no_aa_overlay)
#         y_low3_no_aa, y_high3_no_aa = get_fit_bounds(x_extrap, m3_no_aa, b3_no_aa, cov3_no_aa)
#         ax1.plot(x_extrap, m3_no_aa * x_extrap + b3_no_aa, 'C1--', alpha=0.3, label=None)
#         ax1.fill_between(x_extrap, y_low3_no_aa, y_high3_no_aa, color='C1', alpha=0.1)
#
#         # K4 (no AA) overlay
#         k4_no_aa_overlay = get_numeric(df_no_aa['K4 (s⁻¹)'])
#         ax1.errorbar(x_no_aa_overlay, k4_no_aa_overlay,
#                     yerr=get_error(df_no_aa['K4 (s⁻¹)']), fmt='v',
#                     linewidth=1, markersize=4, alpha=0.3, color='C2', capsize=3, label=None)
#         m4_no_aa, b4_no_aa, cov4_no_aa, *_ = linear_fit(x_no_aa_overlay, k4_no_aa_overlay)
#         y_low4_no_aa, y_high4_no_aa = get_fit_bounds(x_extrap, m4_no_aa, b4_no_aa, cov4_no_aa)
#         ax1.plot(x_extrap, m4_no_aa * x_extrap + b4_no_aa, 'C2--', alpha=0.3, label=None)
#         ax1.fill_between(x_extrap, y_low4_no_aa, y_high4_no_aa, color='C2', alpha=0.1)
#
#         # K1 (no AA) overlay
#         k1_no_aa_overlay = get_numeric(df_no_aa['K1 (s⁻¹)'])
#         ax1.errorbar(x_no_aa_overlay, k1_no_aa_overlay,
#                      yerr=get_error(df_no_aa['K1 (s⁻¹)']), fmt='o',
#                      linewidth=1, markersize=4, alpha=0.3, color='C0', capsize=3, label=None)
#         m1_no_aa, b1_no_aa, cov1_no_aa, *_ = linear_fit(x_no_aa_overlay[:], k1_no_aa_overlay[:])
#         y_low1_no_aa, y_high1_no_aa = get_fit_bounds(x_extrap, m1_no_aa, b1_no_aa, cov1_no_aa)
#         ax1.plot(x_extrap, m1_no_aa * x_extrap + b1_no_aa, 'C0--', alpha=0.3, label=None)
#         ax1.fill_between(x_extrap, y_low1_no_aa, y_high1_no_aa, color='C0', alpha=0.1)
#
#         # K2_light (no AA) overlay
#         k2_no_aa_overlay, k2_no_aa_overlay_err = get_k2_average(df_no_aa)
#         if FIT_K2_LINEAR:
#             ax1.errorbar(x_no_aa_overlay, k2_no_aa_overlay,
#                          yerr=k2_no_aa_overlay_err, fmt='s',
#                          linewidth=1, markersize=4, alpha=0.3, color='C3', capsize=3, label=None)
#             m2_no_aa, b2_no_aa, cov2_no_aa, *_ = linear_fit(x_no_aa_overlay, k2_no_aa_overlay)
#             y_low2_no_aa, y_high2_no_aa = get_fit_bounds(x_extrap, m2_no_aa, b2_no_aa, cov2_no_aa)
#             ax1.plot(x_extrap, m2_no_aa * x_extrap + b2_no_aa, 'C3--', alpha=0.3, label=None)
#             ax1.fill_between(x_extrap, y_low2_no_aa, y_high2_no_aa, color='C3', alpha=0.1)
#         elif PLOT_POWER_DEPENDENT_K2:
#             ax1.errorbar(x_no_aa_overlay, k2_no_aa_overlay,
#                          yerr=k2_no_aa_overlay_err, fmt='s',
#                          linewidth=1, markersize=4, alpha=0.3, color='C3', capsize=3, label=None)
#             k2_val = k2_no_aa_overlay.mean() if not k2_no_aa_overlay.empty else 0.0
#             ax1.axhline(k2_val, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)
#         else:
#             if 'Kr1 (s⁻¹)' in df_no_aa.columns and 'Kr2 (s⁻¹)' in df_no_aa.columns:
#                 kr1_no_aa_val = get_numeric(df_no_aa['Kr1 (s⁻¹)']).mean()
#                 kr2_no_aa_val = get_numeric(df_no_aa['Kr2 (s⁻¹)']).mean()
#                 ax1.axhline(kr1_no_aa_val, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)
#                 ax1.axhline(kr2_no_aa_val, color='C3', linestyle='--', alpha=0.3)
#             else:
#                 k2_no_aa_val = k2_no_aa_overlay.iloc[0] if not k2_no_aa_overlay.empty else 0.0
#                 ax1.axhline(k2_no_aa_val, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)
#
#     # Overlay Thylakoid Non-AA data on the AA plot
#
#     ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
#     ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')
#
#     # ax1.set_xscale('log')
#     # ax1.set_yscale('log')
#     ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
#     ax1.set_xlim(1.5, None)
#
#     # Custom legend without error bars
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
#     ]
#     if not FIT_K2_LINEAR and not PLOT_POWER_DEPENDENT_K2 and 'Kr1 (s⁻¹)' in df_with_aa.columns and 'Kr2 (s⁻¹)' in df_with_aa.columns:
#         legend_elements.extend([
#             Line2D([0], [0], marker=None, color='C3', label=r'$k_{2a}$', linestyle='--', markersize=4),
#             Line2D([0], [0], marker=None, color='C4', label=r'$k_{2b}$', linestyle='--', markersize=4),
#         ])
#     else:
#         legend_elements.append(
#             Line2D([0], [0], marker='s' if (FIT_K2_LINEAR or PLOT_POWER_DEPENDENT_K2) else None, color='C3', label=r'$k_2$', linestyle='None' if (FIT_K2_LINEAR or PLOT_POWER_DEPENDENT_K2) else '--', markersize=4)
#         )
#     legend_elements.extend([
#         Line2D([0], [0], marker='^', color='C1', label=r'$k_3$', linestyle='None', markersize=4),
#         Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4),
#         Line2D([0], [0], marker=None, color='gray', label='Thylakoid (no AA)', linestyle='--', alpha=0.3)
#     ])
#
#     # linear scale
#     # ax1.text(2, 0.55, r'Rates at 2 mmol photons m$^{-2}$ s$^{-1}$:', color='k')
#     # ax1.text(2, 0.4, rf'{k1_extrap:.2g} s$^{{-1}}$', color='C0')
#     # ax1.text(2, 0.25, rf'{k3_extrap:.2g} s$^{{-1}}$', color='C1')
#     # ax1.text(2, 0.1, rf'{k4_extrap:.2g} s$^{{-1}}$', color='C2')
#     # ax1.legend(handles=legend_elements, loc='center left', frameon=False)
#     # log scale
#     ax1.text(2, 0.015, r'Rates at 2 mmol photons m$^{-2}$ s$^{-1}$:', color='k', rotation=26)
#     ax1.text(2, k1_extrap*1.3, rf'{k1_extrap:.2g} s$^{{-1}}$', color='C0', rotation=26)
#     ax1.text(2, k3_extrap*1.5, rf'{k3_extrap:.2g} s$^{{-1}}$', color='C1', rotation=26)
#     ax1.text(2, k4_extrap*1.0, rf'{k4_extrap:.2g} s$^{{-1}}$', color='C2', rotation=26)
#     ax1.legend(handles=legend_elements, loc='lower right', frameon=False)
#
#     print_bic_summary(fits_dict_aa, title="With AA LHCII Linear Fits & Total BIC Summary")
#     sns.despine()
#     plt.tight_layout()
#     plot_file_aa = os.path.join(base_data_dir, 'k_values_vs_power_with_AA.png')
#     plt.savefig(plot_file_aa, dpi=300, bbox_inches='tight')
#     print(f"With AA plot saved to: {plot_file_aa}")
#     plt.show()

# Create unified figure for "without AA" data
if not df_no_aa.empty:
    fig_no_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 60 / 25.4))  # main text 1c
    # fig_no_aa, ax1 = plt.subplots(1, 1, figsize=(80 / 25.4, 50 / 25.4))  # SI light-dependent k2
    fits_dict_no_aa = {}

    x_no_aa = df_no_aa['Power (mE)'].values
    # x_extrap = np.linspace(min(2, x_no_aa.min()), max(1000, x_no_aa.max()), 100)
    x_extrap = np.logspace(0, 3, 100)

    # Plot K3 (no AA)
    k3_no_aa = get_numeric(df_no_aa['K3 (s⁻¹)'])
    k3_plot = ax1.errorbar(x_no_aa[:-1], k3_no_aa[:-1],
                yerr=get_error(df_no_aa['K3 (s⁻¹)'][:-1]), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=3, alpha=1, color='C1', capsize=3)
    # Linear fit for K3
    m3, b3, cov3, r3, rss3, bic3, n3 = linear_fit(x_no_aa[:-2], k3_no_aa[:-2])
    fits_dict_no_aa['k3'] = {'N': n3, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m3, 'b': b3, 'r': r3, 'rss': rss3, 'bic': bic3}
    k3_extrap = m3 * 2 + b3
    y_low3, y_high3 = get_fit_bounds(x_extrap, m3, b3, cov3)
    ax1.plot(x_extrap, m3 * x_extrap + b3, 'C1-', label=None)
    ax1.fill_between(x_extrap, y_low3, y_high3, color='C1', alpha=0.2)

    # Plot K4 (no AA)
    k4_no_aa = get_numeric(df_no_aa['K4 (s⁻¹)'])
    k4_plot = ax1.errorbar(x_no_aa[:-1],  k4_no_aa[:-1],
                yerr=get_error(df_no_aa['K4 (s⁻¹)'][:-1]), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=3, alpha=1, color='C2', capsize=3)
    # Linear fit for K4
    m4, b4, cov4, r4, rss4, bic4, n4 = linear_fit(x_no_aa[:-2], k4_no_aa[:-2])
    fits_dict_no_aa['k4'] = {'N': n4, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m4, 'b': b4, 'r': r4, 'rss': rss4, 'bic': bic4}
    k4_extrap = m4 * 2 + b4
    y_low4, y_high4 = get_fit_bounds(x_extrap, m4, b4, cov4)
    ax1.plot(x_extrap, m4 * x_extrap + b4, 'C2-', label=None)
    ax1.fill_between(x_extrap, y_low4, y_high4, color='C2', alpha=0.2)

    # Plot K1 (no AA)
    k1_no_aa = get_numeric(df_no_aa['K1 (s⁻¹)'])
    k1_plot = ax1.errorbar(x_no_aa[:-1], k1_no_aa[:-1],
                 yerr=get_error(df_no_aa['K1 (s⁻¹)'][:-1]), fmt='o', label=r'$k_1$',
                 linewidth=1, markersize=3, alpha=1, color='C0', capsize=3)
    # Linear fit for K1
    m1, b1, cov1, r1, rss1, bic1, n1 = linear_fit(x_no_aa[:-2], k1_no_aa[:-2])
    fits_dict_no_aa['k1'] = {'N': n1, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m1, 'b': b1, 'r': r1, 'rss': rss1, 'bic': bic1}
    k1_extrap = m1 * 2 + b1
    y_low1, y_high1 = get_fit_bounds(x_extrap, m1, b1, cov1)
    ax1.plot(x_extrap, m1 * x_extrap + b1, 'C0-', label=None)
    ax1.fill_between(x_extrap, y_low1, y_high1, color='C0', alpha=0.2)

    # Plot K2 (no AA)
    k2_no_aa, k2_no_aa_err = get_k2_average(df_no_aa)
    if FIT_K2_LINEAR:
        k2_x_fit = x_no_aa[:-2] if len(x_no_aa) > 2 else x_no_aa
        k2_y_fit = k2_no_aa[:-2] if len(k2_no_aa) > 2 else k2_no_aa
        m2, b2, cov2, r2, rss2, bic2, n2 = linear_fit(k2_x_fit, k2_y_fit)
        fits_dict_no_aa['k2'] = {'N': n2, 'k': 2 if FIT_WITH_INTERCEPT else 1, 'm': m2, 'b': b2, 'r': r2, 'rss': rss2, 'bic': bic2}
        k2_extrap = m2 * 2 + b2
        y_low2, y_high2 = get_fit_bounds(x_extrap, m2, b2, cov2)

        k2_plot = ax1.errorbar(x_no_aa[:-1] if len(x_no_aa) > 2 else x_no_aa,
                               k2_no_aa[:-1] if len(k2_no_aa) > 2 else k2_no_aa,
                               yerr=k2_no_aa_err[:-1] if len(k2_no_aa_err) > 2 else k2_no_aa_err,
                               fmt='s', label=r'$k_2$',
                               linewidth=1, markersize=3, alpha=1, color='C3', capsize=3)
        ax1.plot(x_extrap, m2 * x_extrap + b2, 'C3-', label=None)
        ax1.fill_between(x_extrap, y_low2, y_high2, color='C3', alpha=0.2)
    elif PLOT_POWER_DEPENDENT_K2:
        k2_plot = ax1.errorbar(x_no_aa[:-1] if len(x_no_aa) > 2 else x_no_aa,
                               k2_no_aa[:-1] if len(k2_no_aa) > 2 else k2_no_aa,
                               yerr=0.08,
                               fmt='s', label=r'$k_2$',
                               linewidth=1, markersize=3, alpha=1, color='C3', capsize=3)
        k2_val = k2_no_aa.mean() if not k2_no_aa.empty else 0.0
        ax1.axhline(k2_val, color='C3', linestyle='--', label=r'$k_2$')
    else:
        if 'Kr1 (s⁻¹)' in df_no_aa.columns and 'Kr2 (s⁻¹)' in df_no_aa.columns:
            kr1_val = get_numeric(df_no_aa['Kr1 (s⁻¹)']).mean()
            kr2_val = get_numeric(df_no_aa['Kr2 (s⁻¹)']).mean()
            ax1.axhline(kr1_val, color='C3', linestyle='--', label=r'$k_{2a}$')
            ax1.axhline(kr2_val, color='C4', linestyle='--', label=r'$k_{2b}$')
            ax1.scatter(x_no_aa[:-1], [kr1_val for x in x_no_aa[:-1]], color='C3', marker='s', label=None, s=5)
            ax1.scatter(x_no_aa[:-1], [kr2_val for x in x_no_aa[:-1]], color='C4', marker='s', label=None, s=5)
        else:
            k2_val = k2_no_aa.iloc[0] if not k2_no_aa.empty else 0.0
            ax1.axhline(k2_val, color='C3', linestyle='--', label=r'$k_2$')

    ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
    ax1.set_xlabel(r'Photon flux density (mmol m$^{-2}$ s$^{-1}$)')


    if False:
    # if not df_thylakoid_no_aa.empty:
        x_thy_no_aa = df_thylakoid_no_aa['Power (mE)'].values

        # K3 (Thylakoid no AA) overlay
        k3_thy = get_numeric(df_thylakoid_no_aa['K3 (s⁻¹)'])
        ax1.errorbar(x_thy_no_aa, k3_thy,
                     yerr=get_error(df_thylakoid_no_aa['K3 (s⁻¹)']), fmt='^',
                     linewidth=1, markersize=4, alpha=0.3, color='C1', capsize=3, label=None)
        m3_thy, b3_thy, cov3_thy, *_ = linear_fit(x_thy_no_aa, k3_thy)
        ax1.plot(x_extrap, m3_thy * x_extrap + b3_thy, 'C1--', alpha=0.3, label=None)

        # K4 (Thylakoid no AA) overlay
        k4_thy = get_numeric(df_thylakoid_no_aa['K4 (s⁻¹)'])
        ax1.errorbar(x_thy_no_aa, k4_thy,
                     yerr=get_error(df_thylakoid_no_aa['K4 (s⁻¹)']), fmt='v',
                     linewidth=1, markersize=4, alpha=0.3, color='C2', capsize=3, label=None)
        m4_thy, b4_thy, cov4_thy, *_ = linear_fit(x_thy_no_aa, k4_thy)
        ax1.plot(x_extrap, m4_thy * x_extrap + b4_thy, 'C2--', alpha=0.3, label=None)

        # K1 (Thylakoid no AA) overlay
        k1_thy = get_numeric(df_thylakoid_no_aa['K1 (s⁻¹)'])
        ax1.errorbar(x_thy_no_aa, k1_thy,
                     yerr=get_error(df_thylakoid_no_aa['K1 (s⁻¹)']), fmt='o',
                     linewidth=1, markersize=4, alpha=0.3, color='C0', capsize=3, label=None)
        m1_thy, b1_thy, cov1_thy, *_ = linear_fit(x_thy_no_aa[:-3], k1_thy[:-3])
        ax1.plot(x_extrap, m1_thy * x_extrap + b1_thy, 'C0--', alpha=0.3, label=None)

        # K2 (Thylakoid no AA) overlay
        k2_thy_series, k2_thy_err_series = get_k2_average(df_thylakoid_no_aa)
        if (FIT_K2_LINEAR or PLOT_POWER_DEPENDENT_K2) and not k2_thy_series.empty:
            ax1.errorbar(x_thy_no_aa, k2_thy_series,
                         yerr=k2_thy_err_series, fmt='s',
                         linewidth=1, markersize=4, alpha=0.3, color='C3', capsize=3, label=None)
            m2_thy, b2_thy, cov2_thy, *_ = linear_fit(x_thy_no_aa[:-3] if len(x_thy_no_aa) > 3 else x_thy_no_aa,
                                                      k2_thy_series[:-3] if len(k2_thy_series) > 3 else k2_thy_series)
            y_low2_thy, y_high2_thy = get_fit_bounds(x_extrap, m2_thy, b2_thy, cov2_thy)
            ax1.plot(x_extrap, m2_thy * x_extrap + b2_thy, 'C3--', alpha=0.3, label=None)
        else:
            if 'Kr1 (s⁻¹)' in df_thylakoid_no_aa.columns and 'Kr2 (s⁻¹)' in df_thylakoid_no_aa.columns:
                kr1_thy = get_numeric(df_thylakoid_no_aa['Kr1 (s⁻¹)']).mean()
                kr2_thy = get_numeric(df_thylakoid_no_aa['Kr2 (s⁻¹)']).mean()
                ax1.axhline(kr1_thy, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)
                ax1.axhline(kr2_thy, color='C3', linestyle='--', alpha=0.3)
            else:
                k2_thy = k2_thy_series.iloc[0] if not k2_thy_series.empty else 0.0
                ax1.axhline(k2_thy, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(10, 1000)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Custom legend without error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
    ]
    if not FIT_K2_LINEAR and not PLOT_POWER_DEPENDENT_K2 and 'Kr1 (s⁻¹)' in df_no_aa.columns and 'Kr2 (s⁻¹)' in df_no_aa.columns:
        legend_elements.extend([
            Line2D([0], [0], marker='s', color='C3', label=r'$k_{2a}$', linestyle='--', markersize=4),
            Line2D([0], [0], marker='s', color='C4', label=r'$k_{2b}$', linestyle='--', markersize=4),
        ])
    else:
        legend_elements.append(
            Line2D([0], [0], marker='s' if (FIT_K2_LINEAR or PLOT_POWER_DEPENDENT_K2) else None, color='C3', label=r'$k_2$', linestyle='None' if (FIT_K2_LINEAR or PLOT_POWER_DEPENDENT_K2) else '--', markersize=4)
        )
    legend_elements.extend([
        Line2D([0], [0], marker='^', color='C1', label=r'$k_3$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4),
    ])
    ax1.legend(handles=legend_elements, loc='lower right', frameon=False, ncol=1, bbox_to_anchor=(1, -0.02))

    print_bic_summary(fits_dict_no_aa, title="No AA LHCII Linear Fits & Total BIC Summary")

    # sns.despine()
    plt.tight_layout()
    plot_file_no_aa = os.path.join(base_data_dir, 'k_values_vs_power_no_AA.png')
    plt.savefig(plot_file_no_aa, dpi=300, bbox_inches='tight')
    print(f"No AA plot saved to: {plot_file_no_aa}")
    plt.show()
else:
    print("No data to plot")
