import sys
import os

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

import utils

utils.setup_plotting()

def get_numeric(series):
    if series.dtype == 'O':
        return series.str.split(' ±').str[0].astype(float)
    return series.astype(float)

def get_error(series):
    if series.dtype == 'O':
        # Extract the error part, handle cases where no error is present
        parts = series.str.split(' ±')
        return parts.apply(lambda x: float(x[1]) if len(x) > 1 else 0.0)
    return pd.Series(0.0, index=series.index)

def linear_fit(x, y, weights=None):
    if FIT_WITH_INTERCEPT:
        p, cov = np.polyfit(x, y, 1, w=weights, cov=True)
        return p[0], p[1] if p[1] > 0 else 0, cov
    else:
        # Manual calculation for zero-intercept fit: y = m * x
        x_sq_sum = np.sum(x**2)
        m = np.sum(x * y) / x_sq_sum
        # Residual variance estimate
        resid = y - m * x
        n = len(x)
        sigma2 = np.sum(resid**2) / (n - 1) if n > 1 else 0
        m_var = sigma2 / x_sq_sum
        cov = np.array([[m_var, 0], [0, 0]])
        return m, 0.0, cov

def get_fit_bounds(x_range, m, b, cov):
    y_fit = m * x_range + b
    # Variance propagation for y = mx + b
    # var(y) = x^2 * var(m) + var(b) + 2 * x * cov(m, b)
    var_y = (x_range**2 * cov[0,0]) + (2 * x_range * cov[0,1]) + cov[1,1]
    sig_y = np.sqrt(np.maximum(var_y, 0))
    return np.clip(y_fit - sig_y, 1e-4, None), y_fit + sig_y

def get_asymmetric_error(y, yerr, min_val=1e-3):
    """
    Returns asymmetric error bars [lower_err, upper_err] such that
    y - lower_err >= min_val.
    min_val can be a scalar or an array-like of the same length as y.
    """
    # Ensure min_val is at least 0 to avoid errors in calculation
    min_val_adj = np.maximum(0, min_val)
    lower_limit = np.maximum(0, y - min_val_adj)
    lower_err = np.minimum(yerr, lower_limit)
    upper_err = yerr
    return [lower_err, upper_err]

# Load data
results_dir = os.path.join(project_root, 'results')
use_csv = os.path.exists(os.path.join(results_dir, 'local_analysis_results.csv'))

df_no_aa = pd.DataFrame()
df_with_aa = pd.DataFrame()
df_lhcii_no_aa = pd.DataFrame()

if use_csv:
    csv_path = os.path.join(results_dir, 'local_analysis_results.csv')
    print(f"Loading data from {csv_path}")
    df_all = pd.read_csv(csv_path)
    df_no_aa = df_all[df_all['AA'] == 'No'].copy()
    df_with_aa = df_all[df_all['AA'] == 'Yes'].copy()
else:
    try:
        path = os.path.join(results_dir, '../results/data_no_aa_1q.pkl')
        with open(path, 'rb') as f:
            df_no_aa = pickle.load(f)
    except FileNotFoundError:
        df_no_aa = pd.DataFrame()
        print("data_no_aa.pkl not found in results/")

    try:
        path = os.path.join(results_dir, '../results/data_with_aa_1q.pkl')
        with open(path, 'rb') as f:
            df_with_aa = pickle.load(f)
    except FileNotFoundError:
        df_with_aa = pd.DataFrame()
        print("data_with_aa.pkl not found in results/")

try:
    path = os.path.join(results_dir, 'data_no_aa_lhcii.pkl')
    with open(path, 'rb') as f:
        df_lhcii_no_aa = pickle.load(f)
except FileNotFoundError:
    df_lhcii_no_aa = pd.DataFrame()
    print("data_no_aa_lhcii.pkl not found in results/")

# Base data directory for saving plots (can be customized or loaded from data if available)
# In the original script, it was base_data_dir. We'll use current dir or a default.
base_data_dir = '.' 

# Create figure for "with AA" data: single axis for k1, k3, k4
if not df_with_aa.empty:
    fit_intercept = True  # Set to True to fit with an intercept

    fig_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 60 / 25.4))

    x_aa = df_with_aa['Power (mE)'].values
    x_extrap = np.linspace(min(1, x_aa.min()), max(2, x_aa.max()), 100)

    # Plot K3 (with AA)
    k3_aa = get_numeric(df_with_aa['K3 (s⁻¹)'])
    k3_err = get_error(df_with_aa['K3 (s⁻¹)'])
    k3_min = np.zeros_like(k3_aa)
    if len(k3_min) > 0:
        k3_min[1] = 4e-4
        k3_min[4] = 1e-2
        k3_min[5] = 8e-3
        k3_min[6] = 4e-3
    k3_plot = ax1.errorbar(x_aa, k3_aa,
                yerr=get_asymmetric_error(k3_aa, k3_err, k3_min), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=4, alpha=1, color='C1', capsize=3)
    # Linear fit for K3
    m3, b3, cov3 = linear_fit(x_aa[:5], k3_aa[:5])
    k3_fit = m3 * x_extrap + b3
    k3_extrap = m3 * 2 + b3
    print('b3', b3)
    y_low3, y_high3 = get_fit_bounds(x_extrap, m3, b3, cov3)
    ax1.plot(x_extrap, k3_fit, 'C1-', label=None)
    ax1.fill_between(x_extrap, y_low3, y_high3, color='C1', alpha=0.2)

    # Plot K4 (with AA)
    k4_aa = get_numeric(df_with_aa['K4 (s⁻¹)'])
    k4_err = get_error(df_with_aa['K4 (s⁻¹)'])
    # Define individual minimums for K4 data points
    k4_min = np.zeros_like(k4_aa)
    if len(k4_min) > 0:
        k4_min[1] = 0.025  # Example: different minimum for the first point
        k4_min[4] = 0.15
        k4_min[6] = 0.15

    k4_plot = ax1.errorbar(x_aa,  k4_aa,
                yerr=get_asymmetric_error(k4_aa, k4_err, k4_min), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=4, alpha=1, color='C2', capsize=3)
    # Linear fit for K4
    m4, b4, cov4 = linear_fit(x_aa[:], k4_aa[:])
    k4_fit = m4 * x_extrap + b4
    k4_extrap = m4 * 1 + b4
    y_low4, y_high4 = get_fit_bounds(x_extrap, m4, b4, cov4)
    print('b4', b4)
    ax1.plot(x_extrap, k4_fit, 'C2-', label=None)
    ax1.fill_between(x_extrap, y_low4, y_high4, color='C2', alpha=0.2)

    # Plot K1 (with AA)
    k1_aa = get_numeric(df_with_aa['K1 (s⁻¹)'])
    k1_err = get_error(df_with_aa['K1 (s⁻¹)'])
    k1_plot = ax1.errorbar(x_aa, k1_aa,
                 yerr=get_asymmetric_error(k1_aa, k1_err), fmt='o', label=r'$k_1$',
                 linewidth=1, markersize=4, alpha=1, color='C0', capsize=3)
    # Linear fit for K1
    m1, b1, cov1 = linear_fit(x_aa[1:4], k1_aa[1:4])
    k1_fit = m1 * x_extrap + b1
    k1_extrap = m1 * 2 + b1
    y_low1, y_high1 = get_fit_bounds(x_extrap, m1, b1, cov1)
    print('m1', m1)
    ax1.plot(x_extrap, k1_fit, 'C0-', label=None)
    ax1.fill_between(x_extrap, y_low1, y_high1, color='C0', alpha=0.2)

    k2_aa_series = get_numeric(df_with_aa['K2 (s⁻¹)'])
    k2_aa = k2_aa_series.iloc[0] if not k2_aa_series.empty else 0.0
    k2_plot = ax1.axhline(k2_aa, color='C3', linestyle='--', label=r'$k_2$')
    # ax1.errorbar(x_aa, k2l_aa + k2_aa,
    #              yerr=get_error(df_with_aa['K2_light (s⁻¹)']), fmt='s',
    #              linewidth=1, markersize=4, alpha=1, color='C3', capsize=3, label=None)
    # m2l_no_aa = np.sum(x_aa[:] * k2l_aa[:]) / np.sum(x_aa[:] ** 2)
    # ax1.plot(x_extrap, m2l_no_aa * x_extrap + k2_aa, 'C3--', alpha=1, label=None)

    plot_overlay = True
    # Overlay Non-AA data on the AA plot
    if not df_no_aa.empty and plot_overlay:
        x_no_aa_overlay = df_no_aa['Power (mE)'].values

        # K3 (no AA) overlay
        k3_no_aa_overlay = get_numeric(df_no_aa['K3 (s⁻¹)'])
        k3_err_overlay = get_error(df_no_aa['K3 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k3_no_aa_overlay,
                    yerr=get_asymmetric_error(k3_no_aa_overlay, k3_err_overlay), fmt='^',
                    linewidth=1, markersize=4, alpha=0.2, color='C1', capsize=3, label=None)
        # Linear fit for K3 (no AA) overlay
        m3_no_aa, b3_no_aa, cov3_no_aa = linear_fit(x_no_aa_overlay[:], k3_no_aa_overlay[:])
        k3_fit_no_aa = m3_no_aa * x_extrap + b3
        y_low3_no_aa, y_high3_no_aa = get_fit_bounds(x_extrap, m3_no_aa, b3_no_aa, cov3_no_aa)
        ax1.plot(x_extrap, k3_fit_no_aa, 'C1--', alpha=0.2, label=None)
        # ax1.fill_between(x_extrap, y_low3_no_aa, y_high3_no_aa, color='C1', alpha=0.1)

        # K4 (no AA) overlay
        k4_no_aa_overlay = get_numeric(df_no_aa['K4 (s⁻¹)'])
        k4_err_overlay = get_error(df_no_aa['K4 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k4_no_aa_overlay,
                    yerr=get_asymmetric_error(k4_no_aa_overlay, k4_err_overlay), fmt='v',
                    linewidth=1, markersize=4, alpha=0.2, color='C2', capsize=3, label=None)
        # Linear fit for K4 (no AA) overlay
        m4_no_aa, b4_no_aa, cov4_no_aa = linear_fit(x_no_aa_overlay[:], k4_no_aa_overlay[:])
        k4_fit_no_aa = m4_no_aa * x_extrap + b4
        y_low4_no_aa, y_high4_no_aa = get_fit_bounds(x_extrap, m4_no_aa, b4_no_aa, cov4_no_aa)
        ax1.plot(x_extrap, k4_fit_no_aa, 'C2--', alpha=0.2, label=None)
        # ax1.fill_between(x_extrap, y_low4_no_aa, y_high4_no_aa, color='C2', alpha=0.1)

        # K1 (no AA) overlay
        k1_no_aa_overlay = get_numeric(df_no_aa['K1 (s⁻¹)'])
        k1_err_overlay = get_error(df_no_aa['K1 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k1_no_aa_overlay,
                     yerr=get_asymmetric_error(k1_no_aa_overlay, k1_err_overlay), fmt='o',
                     linewidth=1, markersize=4, alpha=0.2, color='C0', capsize=3, label=None)
        # Linear fit for K1 (no AA) overlay
        m1_no_aa, b1_no_aa, cov1_no_aa = linear_fit(x_no_aa_overlay[:5], k1_no_aa_overlay[:5])
        k1_fit_no_aa = m1_no_aa * x_extrap + b1_no_aa
        y_low1_no_aa, y_high1_no_aa = get_fit_bounds(x_extrap, m1_no_aa, b1_no_aa, cov1_no_aa)
        ax1.plot(x_extrap, k1_fit_no_aa, 'C0--', alpha=0.2, label=None)
        # ax1.fill_between(x_extrap, y_low1_no_aa, y_high1_no_aa, color='C0', alpha=0.1)

        # K2_light (no AA) overlay
        k2_no_aa_series = get_numeric(df_no_aa['K2 (s⁻¹)'])
        k2_no_aa = k2_no_aa_series.iloc[0] if not k2_no_aa_series.empty else 0.0
        ax1.axhline(k2_no_aa, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)
        # k2l_no_aa_overlay = get_numeric(df_no_aa['K2_light (s⁻¹)'])
        # ax1.errorbar(x_no_aa_overlay, k2l_no_aa_overlay + k2_no_aa,
        #              yerr=get_error(df_no_aa['K2_light (s⁻¹)']), fmt='s',
        #              linewidth=1, markersize=4, alpha=0.3, color='C3', capsize=3, label=None)
        # m2l_no_aa = np.sum(x_no_aa_overlay[:] * k2l_no_aa_overlay[:]) / np.sum(x_no_aa_overlay[:]**2)
        # ax1.plot(x_extrap, m2l_no_aa * x_extrap + k2_no_aa, 'C3--', alpha=0.3, label=None)

    ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlim(1.8, None)
    ax1.set_ylim(2e-4, None)

    # Custom legend without error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
        Line2D([0], [0], marker=None, color='C3', label=r'$k_2$', linestyle='--', markersize=4),
        Line2D([0], [0], marker='^', color='C1', label=r'$k_3$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4)
    ]

    # linear scale
    # ax1.text(2, 0.55, r'Rates at 2 mmol photons m$^{-2}$ s$^{-1}$:', color='k')
    # ax1.text(2, 0.4, rf'{k1_extrap:.2g} s$^{{-1}}$', color='C0')
    # ax1.text(2, 0.25, rf'{k3_extrap:.2g} s$^{{-1}}$', color='C1')
    # ax1.text(2, 0.1, rf'{k4_extrap:.2g} s$^{{-1}}$', color='C2')
    # ax1.legend(handles=legend_elements, loc='center left', frameon=False)
    # log scale
    ax1.text(1.9, 0.25, r'Rates at 2 mmol photons m$^{-2}$ s$^{-1}$:', color='k', fontsize=6)
    ax1.text(2, k1_extrap*1.4, rf'{k1_extrap:.2g} s$^{{-1}}$', color='C0', fontsize=6)
    ax1.text(2, k3_extrap*0.4, rf'{k3_extrap:.2g} s$^{{-1}}$', color='C1', fontsize=6)
    ax1.text(2, k4_extrap*0.4, rf'{k4_extrap:.2g} s$^{{-1}}$', color='C2', fontsize=6)
    ax1.legend(handles=legend_elements, loc='lower right', frameon=False, bbox_to_anchor=(0.9, 0))

    sns.despine()
    plt.tight_layout()
    plot_file_aa = os.path.join(base_data_dir, 'k_values_vs_power_with_AA.png')
    # plt.savefig(plot_file_aa, dpi=300, bbox_inches='tight')
    print(f"With AA plot saved to: {plot_file_aa}")
    # plt.show()

# Create unified figure for "without AA" data
if not df_no_aa.empty:
    fit_intercept = True  # Set to True to fit with an intercept

    fig_no_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 50 / 25.4))

    x_no_aa = df_no_aa['Power (mE)'].values
    x_extrap = np.linspace(min(2, x_no_aa.min()), max(2, x_no_aa.max()), 100)

    # Plot K3 (no AA)
    k3_no_aa = get_numeric(df_no_aa['K3 (s⁻¹)'])
    k3_err_no_aa = get_error(df_no_aa['K3 (s⁻¹)'])
    k3_plot = ax1.errorbar(x_no_aa, k3_no_aa,
                yerr=get_asymmetric_error(k3_no_aa, k3_err_no_aa), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=4, alpha=1, color='C1', capsize=3)
    # Linear fit for K3
    m3, b3, cov3 = linear_fit(x_no_aa[:3], k3_no_aa[:3])
    k3_fit = m3 * x_extrap + b3
    y_low3, y_high3 = get_fit_bounds(x_extrap, m3, b3, cov3)
    ax1.plot(x_extrap, k3_fit, 'C1-', label=None)
    ax1.fill_between(x_extrap, y_low3, y_high3, color='C1', alpha=0.2)

    # Plot K4 (no AA)
    k4_no_aa = get_numeric(df_no_aa['K4 (s⁻¹)'])
    k4_err_no_aa = get_error(df_no_aa['K4 (s⁻¹)'])
    k4_plot = ax1.errorbar(x_no_aa,  k4_no_aa,
                yerr=get_asymmetric_error(k4_no_aa, k4_err_no_aa), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=4, alpha=1, color='C2', capsize=3)
    # Linear fit for K4
    m4, b4, cov4 = linear_fit(x_no_aa[:3], k4_no_aa[:3])
    k4_fit = m4 * x_extrap + b4
    k4_extrap = m4
    y_low4, y_high4 = get_fit_bounds(x_extrap, m4, b4, cov4)
    ax1.plot(x_extrap, k4_fit, 'C2-', label=None)
    ax1.fill_between(x_extrap, y_low4, y_high4, color='C2', alpha=0.2)

    # Plot K1 (no AA)
    k1_no_aa = get_numeric(df_no_aa['K1 (s⁻¹)'])
    k1_err_no_aa = get_error(df_no_aa['K1 (s⁻¹)'])
    k1_plot = ax1.errorbar(x_no_aa, k1_no_aa,
                 yerr=get_asymmetric_error(k1_no_aa, k1_err_no_aa), fmt='o', label=r'$k_1$',
                 linewidth=1, markersize=4, alpha=1, color='C0', capsize=3)
    # Linear fit for K1
    m1, b1, cov1 = linear_fit(x_no_aa[:5], k1_no_aa[:5])
    k1_fit = m1 * x_extrap + b1
    k1_extrap = m1 * 2 + b1
    y_low1, y_high1 = get_fit_bounds(x_extrap, m1, b1, cov1)
    ax1.plot(x_extrap, k1_fit, 'C0-', label=None)
    ax1.fill_between(x_extrap, y_low1, y_high1, color='C0', alpha=0.2)

    # Plot K2_light (no AA)
    k2_no_aa_series = get_numeric(df_no_aa['K2 (s⁻¹)'])
    k2_no_aa = k2_no_aa_series.iloc[0] if not k2_no_aa_series.empty else 0.0
    ax1.axhline(k2_no_aa, color='C3', linestyle='--', label=r'$k_2$')
    # k2l_no_aa = get_numeric(df_no_aa['K2_light (s⁻¹)'])
    # k2_plot = ax1.errorbar(x_no_aa, k2l_no_aa + k2_no_aa,
    #              yerr=get_error(df_no_aa['K2_light (s⁻¹)']), fmt='s', label=r'$k_{2}$',
    #              linewidth=1, markersize=4, alpha=1, color='C3', capsize=3)
    # # Linear fit for K2_light with zero intercept
    # m2l = np.sum(x_no_aa[:] * k2l_no_aa[:]) / np.sum(x_no_aa[:]**2)
    # ax1.plot(x_extrap, m2l * x_extrap + k2_no_aa, 'C3--', label=None)

    # Overlay LHCII data
    if not df_lhcii_no_aa.empty:
        x_lhcii = df_lhcii_no_aa['Power (mE)'].values
        
        # K3 LHCII
        k3_lhcii = get_numeric(df_lhcii_no_aa['K3 (s⁻¹)'])[:-2]
        k3_err_lhcii = get_error(df_lhcii_no_aa['K3 (s⁻¹)'])[:-2]
        ax1.errorbar(x_lhcii[:-2], k3_lhcii, yerr=get_asymmetric_error(k3_lhcii, k3_err_lhcii),
                     fmt='^', linewidth=1, markersize=4, alpha=0.2, color='C1', capsize=3)
        m3_lhcii, b3_lhcii, cov3_lhcii = linear_fit(x_lhcii[:-2], k3_lhcii)
        ax1.plot(x_extrap, m3_lhcii * x_extrap + b3_lhcii, 'C1--', alpha=0.2)
        
        # K4 LHCII
        k4_lhcii = get_numeric(df_lhcii_no_aa['K4 (s⁻¹)'])[:-2]
        k4_err_lhcii = get_error(df_lhcii_no_aa['K4 (s⁻¹)'])[:-2]
        ax1.errorbar(x_lhcii[:-2], k4_lhcii, yerr=get_asymmetric_error(k4_lhcii, k4_err_lhcii),
                     fmt='v', linewidth=1, markersize=4, alpha=0.2, color='C2', capsize=3)
        m4_lhcii, b4_lhcii, cov4_lhcii = linear_fit(x_lhcii[:-2], k4_lhcii)
        ax1.plot(x_extrap, m4_lhcii * x_extrap + b4_lhcii, 'C2--', alpha=0.2)

        # K1 LHCII
        k1_lhcii = get_numeric(df_lhcii_no_aa['K1 (s⁻¹)'])[:-2]
        k1_err_lhcii = get_error(df_lhcii_no_aa['K1 (s⁻¹)'])[:-2]
        ax1.errorbar(x_lhcii[:-2], k1_lhcii, yerr=get_asymmetric_error(k1_lhcii, k1_err_lhcii),
                     fmt='o', linewidth=1, markersize=4, alpha=0.2, color='C0', capsize=3)
        m1_lhcii, b1_lhcii, cov1_lhcii = linear_fit(x_lhcii[:-2], k1_lhcii)
        ax1.plot(x_extrap, m1_lhcii * x_extrap + b1_lhcii, 'C0--', alpha=0.2)

        # K2 LHCII
        k2a_lhcii_series = get_numeric(df_lhcii_no_aa['Kr1 (s⁻¹)'])
        k2a_lhcii = k2a_lhcii_series.iloc[0] if not k2a_lhcii_series.empty else 0.0
        k2b_lhcii_series = get_numeric(df_lhcii_no_aa['Kr2 (s⁻¹)'])
        k2b_lhcii = k2b_lhcii_series.iloc[0] if not k2b_lhcii_series.empty else 0.0
        f_lhcii_series = get_numeric(df_lhcii_no_aa['f'])
        f_lhcii = f_lhcii_series.iloc[0] if not f_lhcii_series.empty else 0.0
        k2_lhcii = 1 / (f_lhcii / k2a_lhcii + (1 - f_lhcii) / k2b_lhcii)
        ax1.axhline(k2_lhcii, color='C3', linestyle='--', alpha=0.2)

    ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')

    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(15, None)
    
    # Custom legend without error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
        Line2D([0], [0], marker=None, color='C3', label=r'$k_2$', linestyle='--', markersize=4),
        Line2D([0], [0], marker='^', color='C1', label=r'$k_3$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4)
    ]
    ax1.legend(handles=legend_elements, loc='best', frameon=False)

    sns.despine()
    plt.tight_layout()
    plot_file_no_aa = os.path.join(base_data_dir, 'k_values_vs_power_no_AA.png')
    # plt.savefig(plot_file_no_aa, dpi=300, bbox_inches='tight')
    print(f"No AA plot saved to: {plot_file_no_aa}")
    plt.show()
else:
    print("No data to plot")
