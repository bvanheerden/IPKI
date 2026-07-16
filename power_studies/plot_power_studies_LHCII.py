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
        return p[0], p[1], cov
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
    return y_fit - sig_y, y_fit + sig_y


results_dir = os.path.join(project_root, 'results')
df_no_aa = pd.DataFrame()
df_with_aa = pd.DataFrame()
df_thylakoid_no_aa = pd.DataFrame()

try:
    path = os.path.join(results_dir, 'data_no_aa_lhcii.pkl')
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
    path = os.path.join(results_dir, 'data_no_aa.pkl')
    with open(path, 'rb') as f:
        df_thylakoid_no_aa = pickle.load(f)
except FileNotFoundError:
    print("data_no_aa.pkl not found in results/")

# Base data directory for saving plots (can be customized or loaded from data if available)
# In the original script, it was base_data_dir. We'll use current dir or a default.
base_data_dir = '.' 

# Create figure for "with AA" data: single axis for k1, k3, k4
if not df_with_aa.empty:
    print('hello')
    fig_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 70 / 25.4))

    x_aa = df_with_aa['Power (mE)'].values
    x_extrap = np.linspace(min(2, x_aa.min()), max(2, x_aa.max()), 100)

    # Plot K3 (with AA)
    k3_aa = get_numeric(df_with_aa['K3 (s⁻¹)'])
    k3_plot = ax1.errorbar(x_aa, k3_aa,
                yerr=get_error(df_with_aa['K3 (s⁻¹)']), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=4, alpha=1, color='C1', capsize=3)
    # Linear fit for K3
    m3, b3, cov3 = linear_fit(x_aa, k3_aa)
    print('m3', m3, 'b3', b3)
    k3_extrap = m3 * 2 + b3
    y_low3, y_high3 = get_fit_bounds(x_extrap, m3, b3, cov3)
    ax1.plot(x_extrap, m3 * x_extrap + b3, 'C1--', label=None)
    ax1.fill_between(x_extrap, y_low3, y_high3, color='C1', alpha=0.2)
    # Linear fit with intercept
    # model = np.polynomial.Polynomial.fit(x_aa[:-1], k3_aa[:-1], deg=1)
    # ax1.plot(x_extrap, model(x_extrap), 'C1--', label=None)

    # Plot K4 (with AA)
    k4_aa = get_numeric(df_with_aa['K4 (s⁻¹)'])
    k4_plot = ax1.errorbar(x_aa,  k4_aa,
                yerr=get_error(df_with_aa['K4 (s⁻¹)']), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=4, alpha=1, color='C2', capsize=3)
    # Linear fit for K4
    m4, b4, cov4 = linear_fit(x_aa, k4_aa)
    print('m4', m4, 'b4', b4)
    k4_extrap = m4 * 2 + b4
    y_low4, y_high4 = get_fit_bounds(x_extrap, m4, b4, cov4)
    ax1.plot(x_extrap, m4 * x_extrap + b4, 'C2--', label=None)
    ax1.fill_between(x_extrap, y_low4, y_high4, color='C2', alpha=0.2)

    # Plot K1 (with AA)
    k1_aa = get_numeric(df_with_aa['K1 (s⁻¹)'])
    k1_plot = ax1.errorbar(x_aa, k1_aa,
                 yerr=get_error(df_with_aa['K1 (s⁻¹)']), fmt='o', label=r'$k_1$',
                 linewidth=1, markersize=4, alpha=1, color='C0', capsize=3)
    # Linear fit for K1
    m1, b1, cov1 = linear_fit(x_aa, k1_aa)
    print('m1', m1, 'b1', b1)
    k1_extrap = m1 * 2 + b1
    y_low1, y_high1 = get_fit_bounds(x_extrap, m1, b1, cov1)
    ax1.plot(x_extrap, m1 * x_extrap + b1, 'C0--', label=None)
    ax1.fill_between(x_extrap, y_low1, y_high1, color='C0', alpha=0.2)

    k2_aa_series = get_numeric(df_with_aa['Kr1 (s⁻¹)'])
    k2_aa = k2_aa_series.iloc[0] if not k2_aa_series.empty else 0.0
    k2_plot = ax1.axhline(k2_aa, color='C3', linestyle='--', label=r'$k_2$')
    k2l_aa = get_numeric(df_with_aa['K2_light (s⁻¹)'])
    # ax1.errorbar(x_aa, k2l_aa + k2_aa,
    #              yerr=get_error(df_with_aa['K2_light (s⁻¹)']), fmt='s',
    #              linewidth=1, markersize=4, alpha=1, color='C3', capsize=3, label=None)
    # m2l_no_aa = np.sum(x_aa[:] * k2l_aa[:]) / np.sum(x_aa[:] ** 2)
    # ax1.plot(x_extrap, m2l_no_aa * x_extrap + k2_aa, 'C3--', alpha=1, label=None)

    plot_overlay = False
    # Overlay Non-AA data on the AA plot
    if not df_no_aa.empty and plot_overlay:
        x_no_aa_overlay = df_no_aa['Power (mE)'].values

        # K3 (no AA) overlay
        k3_no_aa_overlay = get_numeric(df_no_aa['K3 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k3_no_aa_overlay,
                    yerr=get_error(df_no_aa['K3 (s⁻¹)']), fmt='^',
                    linewidth=1, markersize=4, alpha=0.3, color='C1', capsize=3, label=None)
        m3_no_aa, b3_no_aa, cov3_no_aa = linear_fit(x_no_aa_overlay, k3_no_aa_overlay)
        y_low3_no_aa, y_high3_no_aa = get_fit_bounds(x_extrap, m3_no_aa, b3_no_aa, cov3_no_aa)
        ax1.plot(x_extrap, m3_no_aa * x_extrap + b3_no_aa, 'C1--', alpha=0.3, label=None)
        ax1.fill_between(x_extrap, y_low3_no_aa, y_high3_no_aa, color='C1', alpha=0.1)

        # K4 (no AA) overlay
        k4_no_aa_overlay = get_numeric(df_no_aa['K4 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k4_no_aa_overlay,
                    yerr=get_error(df_no_aa['K4 (s⁻¹)']), fmt='v',
                    linewidth=1, markersize=4, alpha=0.3, color='C2', capsize=3, label=None)
        m4_no_aa, b4_no_aa, cov4_no_aa = linear_fit(x_no_aa_overlay, k4_no_aa_overlay)
        y_low4_no_aa, y_high4_no_aa = get_fit_bounds(x_extrap, m4_no_aa, b4_no_aa, cov4_no_aa)
        ax1.plot(x_extrap, m4_no_aa * x_extrap + b4_no_aa, 'C2--', alpha=0.3, label=None)
        ax1.fill_between(x_extrap, y_low4_no_aa, y_high4_no_aa, color='C2', alpha=0.1)

        # K1 (no AA) overlay
        k1_no_aa_overlay = get_numeric(df_no_aa['K1 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k1_no_aa_overlay,
                     yerr=get_error(df_no_aa['K1 (s⁻¹)']), fmt='o',
                     linewidth=1, markersize=4, alpha=0.3, color='C0', capsize=3, label=None)
        m1_no_aa, b1_no_aa, cov1_no_aa = linear_fit(x_no_aa_overlay[:], k1_no_aa_overlay[:])
        y_low1_no_aa, y_high1_no_aa = get_fit_bounds(x_extrap, m1_no_aa, b1_no_aa, cov1_no_aa)
        ax1.plot(x_extrap, m1_no_aa * x_extrap + b1_no_aa, 'C0--', alpha=0.3, label=None)
        ax1.fill_between(x_extrap, y_low1_no_aa, y_high1_no_aa, color='C0', alpha=0.1)

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

    # Overlay Thylakoid Non-AA data on the AA plot

    ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')

    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlim(1.5, None)
    
    # Custom legend without error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
        Line2D([0], [0], marker=None, color='C3', label=r'$k_2$', linestyle='--', markersize=4),
        Line2D([0], [0], marker='^', color='C1', label=r'$k_3$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4),
        Line2D([0], [0], marker=None, color='gray', label='Thylakoid (no AA)', linestyle='--', alpha=0.3)
    ]

    # linear scale
    # ax1.text(2, 0.55, r'Rates at 2 mmol photons m$^{-2}$ s$^{-1}$:', color='k')
    # ax1.text(2, 0.4, rf'{k1_extrap:.2g} s$^{{-1}}$', color='C0')
    # ax1.text(2, 0.25, rf'{k3_extrap:.2g} s$^{{-1}}$', color='C1')
    # ax1.text(2, 0.1, rf'{k4_extrap:.2g} s$^{{-1}}$', color='C2')
    # ax1.legend(handles=legend_elements, loc='center left', frameon=False)
    # log scale
    ax1.text(2, 0.015, r'Rates at 2 mmol photons m$^{-2}$ s$^{-1}$:', color='k', rotation=26)
    ax1.text(2, k1_extrap*1.3, rf'{k1_extrap:.2g} s$^{{-1}}$', color='C0', rotation=26)
    ax1.text(2, k3_extrap*1.5, rf'{k3_extrap:.2g} s$^{{-1}}$', color='C1', rotation=26)
    ax1.text(2, k4_extrap*1.0, rf'{k4_extrap:.2g} s$^{{-1}}$', color='C2', rotation=26)
    ax1.legend(handles=legend_elements, loc='lower right', frameon=False)

    sns.despine()
    plt.tight_layout()
    plot_file_aa = os.path.join(base_data_dir, 'k_values_vs_power_with_AA.png')
    plt.savefig(plot_file_aa, dpi=300, bbox_inches='tight')
    print(f"With AA plot saved to: {plot_file_aa}")
    plt.show()

# Create unified figure for "without AA" data
if not df_no_aa.empty:
    fig_no_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 60 / 25.4))

    x_no_aa = df_no_aa['Power (mE)'].values
    x_extrap = np.linspace(min(2, x_no_aa.min()), max(1000, x_no_aa.max()), 100)

    # Plot K3 (no AA)
    k3_no_aa = get_numeric(df_no_aa['K3 (s⁻¹)'])
    k3_plot = ax1.errorbar(x_no_aa[:-1], k3_no_aa[:-1],
                yerr=get_error(df_no_aa['K3 (s⁻¹)'][:-1]), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=3, alpha=1, color='C1', capsize=3)
    # Linear fit for K3
    m3, b3, cov3 = linear_fit(x_no_aa[:-1], k3_no_aa[:-1])
    k3_extrap = m3 * 2 + b3
    print(m3, b3)
    y_low3, y_high3 = get_fit_bounds(x_extrap, m3, b3, cov3)
    ax1.plot(x_extrap, m3 * x_extrap + b3, 'C1-', label=None)
    ax1.fill_between(x_extrap, y_low3, y_high3, color='C1', alpha=0.2)

    # Plot K4 (no AA)
    k4_no_aa = get_numeric(df_no_aa['K4 (s⁻¹)'])
    k4_plot = ax1.errorbar(x_no_aa[:-1],  k4_no_aa[:-1],
                yerr=get_error(df_no_aa['K4 (s⁻¹)'][:-1]), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=3, alpha=1, color='C2', capsize=3)
    # Linear fit for K4
    m4, b4, cov4 = linear_fit(x_no_aa[:-1], k4_no_aa[:-1])
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
    m1, b1, cov1 = linear_fit(x_no_aa[:-1], k1_no_aa[:-1])
    k1_extrap = m1 * 2 + b1
    y_low1, y_high1 = get_fit_bounds(x_extrap, m1, b1, cov1)
    ax1.plot(x_extrap, m1 * x_extrap + b1, 'C0-', label=None)
    ax1.fill_between(x_extrap, y_low1, y_high1, color='C0', alpha=0.2)

    # Plot K2_light (no AA)
    k2a_no_aa_series = get_numeric(df_no_aa['Kr1 (s⁻¹)'])
    k2a_no_aa = k2a_no_aa_series.iloc[0] if not k2a_no_aa_series.empty else 0.0
    ax1.axhline(k2a_no_aa, color='C3', linestyle='--', label=r'$k_{2a}$')

    k2b_no_aa_series = get_numeric(df_no_aa['Kr2 (s⁻¹)'])
    k2b_no_aa = k2b_no_aa_series.iloc[0] if not k2b_no_aa_series.empty else 0.0
    ax1.axhline(k2b_no_aa, color='C4', linestyle='--', label=r'$k_{2b}$')
    # k2l_no_aa = get_numeric(df_no_aa['K2_light (s⁻¹)'])
    # k2_plot = ax1.errorbar(x_no_aa, k2l_no_aa + k2_no_aa,
    #              yerr=get_error(df_no_aa['K2_light (s⁻¹)']), fmt='s', label=r'$k_{2}$',
    #              linewidth=1, markersize=4, alpha=1, color='C3', capsize=3)
    # # Linear fit for K2_light with zero intercept
    # m2l = np.sum(x_no_aa[:] * k2l_no_aa[:]) / np.sum(x_no_aa[:]**2)
    # ax1.plot(x_extrap, m2l * x_extrap + k2_no_aa, 'C3--', label=None)

    ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')


    if False:
    # if not df_thylakoid_no_aa.empty:
        x_thy_no_aa = df_thylakoid_no_aa['Power (mE)'].values

        # K3 (Thylakoid no AA) overlay
        k3_thy = get_numeric(df_thylakoid_no_aa['K3 (s⁻¹)'])
        ax1.errorbar(x_thy_no_aa, k3_thy,
                     yerr=get_error(df_thylakoid_no_aa['K3 (s⁻¹)']), fmt='^',
                     linewidth=1, markersize=4, alpha=0.3, color='C1', capsize=3, label=None)
        m3_thy, b3_thy = linear_fit(x_thy_no_aa, k3_thy)
        ax1.plot(x_extrap, m3_thy * x_extrap + b3_thy, 'C1--', alpha=0.3, label=None)

        # K4 (Thylakoid no AA) overlay
        k4_thy = get_numeric(df_thylakoid_no_aa['K4 (s⁻¹)'])
        ax1.errorbar(x_thy_no_aa, k4_thy,
                     yerr=get_error(df_thylakoid_no_aa['K4 (s⁻¹)']), fmt='v',
                     linewidth=1, markersize=4, alpha=0.3, color='C2', capsize=3, label=None)
        m4_thy, b4_thy = linear_fit(x_thy_no_aa, k4_thy)
        ax1.plot(x_extrap, m4_thy * x_extrap + b4_thy, 'C2--', alpha=0.3, label=None)

        # K1 (Thylakoid no AA) overlay
        k1_thy = get_numeric(df_thylakoid_no_aa['K1 (s⁻¹)'])
        ax1.errorbar(x_thy_no_aa, k1_thy,
                     yerr=get_error(df_thylakoid_no_aa['K1 (s⁻¹)']), fmt='o',
                     linewidth=1, markersize=4, alpha=0.3, color='C0', capsize=3, label=None)
        m1_thy, b1_thy = linear_fit(x_thy_no_aa[:-3], k1_thy[:-3])
        ax1.plot(x_extrap, m1_thy * x_extrap + b1_thy, 'C0--', alpha=0.3, label=None)

        # K2 (Thylakoid no AA) overlay
        k2_thy_series = get_numeric(df_thylakoid_no_aa['K2 (s⁻¹)'])
        k2_thy = k2_thy_series.iloc[0] if not k2_thy_series.empty else 0.0
        ax1.axhline(k2_thy, color='C3', linestyle='--', label=r'$k_2$', alpha=0.3)

    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(60, 1000)
    
    # Custom legend without error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='^', color='C1', label=r'$k_1$', linestyle='None', markersize=4),
        Line2D([0], [0], marker=None, color='C3', label=r'$k_{2a}$', linestyle='--', markersize=4),
        Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4),
        Line2D([0], [0], marker=None, color='C4', label=r'$k_{2b}$', linestyle='--', markersize=4),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=False, ncol=3)#, bbox_to_anchor=(0, 0.95))

    sns.despine()
    plt.tight_layout()
    plot_file_no_aa = os.path.join(base_data_dir, 'k_values_vs_power_no_AA.png')
    plt.savefig(plot_file_no_aa, dpi=300, bbox_inches='tight')
    print(f"No AA plot saved to: {plot_file_no_aa}")
    plt.show()
else:
    print("No data to plot")
