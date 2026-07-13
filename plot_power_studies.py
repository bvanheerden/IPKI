import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# Enable LaTeX rendering globally
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

sns.set_palette("deep")

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

# Load data
use_csv = os.path.exists('local_analysis_results.csv')

df_no_aa = pd.DataFrame()
df_with_aa = pd.DataFrame()

if use_csv:
    print("Loading data from local_analysis_results.csv")
    df_all = pd.read_csv('local_analysis_results.csv')
    df_no_aa = df_all[df_all['AA'] == 'No'].copy()
    df_with_aa = df_all[df_all['AA'] == 'Yes'].copy()
else:
    try:
        with open('data_no_aa.pkl', 'rb') as f:
            df_no_aa = pickle.load(f)
    except FileNotFoundError:
        df_no_aa = pd.DataFrame()
        print("data_no_aa_k2.pkl not found")

    try:
        with open('data_with_aa.pkl', 'rb') as f:
            df_with_aa = pickle.load(f)
    except FileNotFoundError:
        df_with_aa = pd.DataFrame()
        print("data_with_aa_k2.pkl not found")

# Base data directory for saving plots (can be customized or loaded from data if available)
# In the original script, it was base_data_dir. We'll use current dir or a default.
base_data_dir = '.' 

# Create figure for "with AA" data: single axis for k1, k3, k4
if not df_with_aa.empty:
    fig_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 70 / 25.4))

    x_aa = df_with_aa['Power (mE)'].values
    x_extrap = np.linspace(min(2, x_aa.min()), max(2, x_aa.max()), 100)

    # Plot K3 (with AA)
    k3_aa = get_numeric(df_with_aa['K3 (s⁻¹)'])
    k3_plot = ax1.errorbar(x_aa, k3_aa,
                yerr=get_error(df_with_aa['K3 (s⁻¹)']), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=4, alpha=1, color='C1', capsize=3)
    # Linear fit for K3 with zero intercept
    m3 = np.sum(x_aa[:] * k3_aa[:]) / np.sum(x_aa[:]**2)
    print('m3', m3)
    k3_extrap = m3 * 2
    ax1.plot(x_extrap, m3 * x_extrap, 'C1--', label=None)
    # Linear fit with intercept
    # model = np.polynomial.Polynomial.fit(x_aa[:-1], k3_aa[:-1], deg=1)
    # ax1.plot(x_extrap, model(x_extrap), 'C1--', label=None)

    # Plot K4 (with AA)
    k4_aa = get_numeric(df_with_aa['K4 (s⁻¹)'])
    k4_plot = ax1.errorbar(x_aa,  k4_aa,
                yerr=get_error(df_with_aa['K4 (s⁻¹)']), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=4, alpha=1, color='C2', capsize=3)
    # Linear fit for K4 with zero intercept
    m4 = np.sum(x_aa[:] * k4_aa[:]) / np.sum(x_aa[:]**2)
    print('m4', m4)
    k4_extrap = m4
    ax1.plot(x_extrap, m4 * x_extrap, 'C2--', label=None)

    # Plot K1 (with AA)
    k1_aa = get_numeric(df_with_aa['K1 (s⁻¹)'])
    k1_plot = ax1.errorbar(x_aa, k1_aa,
                 yerr=get_error(df_with_aa['K1 (s⁻¹)']), fmt='o', label=r'$k_1$',
                 linewidth=1, markersize=4, alpha=1, color='C0', capsize=3)
    # Linear fit for K1 with zero intercept
    m1 = np.sum(x_aa[:] * k1_aa[:]) / np.sum(x_aa[:]**2)
    print('m1', m1)
    k1_extrap = m1 * 2
    ax1.plot(x_extrap, m1 * x_extrap, 'C0--', label=None)

    k2_aa_series = get_numeric(df_with_aa['K2 (s⁻¹)'])
    k2_aa = k2_aa_series.iloc[0] if not k2_aa_series.empty else 0.0
    k2_plot = ax1.axhline(k2_aa, color='C3', linestyle='--', label=r'$k_2$')
    k2l_aa = get_numeric(df_with_aa['K2_light (s⁻¹)'])
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
        ax1.errorbar(x_no_aa_overlay, k3_no_aa_overlay,
                    yerr=get_error(df_no_aa['K3 (s⁻¹)']), fmt='^',
                    linewidth=1, markersize=4, alpha=0.3, color='C1', capsize=3, label=None)
        m3_no_aa = np.sum(x_no_aa_overlay[:] * k3_no_aa_overlay[:]) / np.sum(x_no_aa_overlay[:]**2)
        ax1.plot(x_extrap, m3_no_aa * x_extrap, 'C1--', alpha=0.3, label=None)

        # K4 (no AA) overlay
        k4_no_aa_overlay = get_numeric(df_no_aa['K4 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k4_no_aa_overlay,
                    yerr=get_error(df_no_aa['K4 (s⁻¹)']), fmt='v',
                    linewidth=1, markersize=4, alpha=0.3, color='C2', capsize=3, label=None)
        m4_no_aa = np.sum(x_no_aa_overlay[:] * k4_no_aa_overlay[:]) / np.sum(x_no_aa_overlay[:]**2)
        ax1.plot(x_extrap, m4_no_aa * x_extrap, 'C2--', alpha=0.3, label=None)

        # K1 (no AA) overlay
        k1_no_aa_overlay = get_numeric(df_no_aa['K1 (s⁻¹)'])
        ax1.errorbar(x_no_aa_overlay, k1_no_aa_overlay,
                     yerr=get_error(df_no_aa['K1 (s⁻¹)']), fmt='o',
                     linewidth=1, markersize=4, alpha=0.3, color='C0', capsize=3, label=None)
        m1_no_aa = np.sum(x_no_aa_overlay[:-3] * k1_no_aa_overlay[:-3]) / np.sum(x_no_aa_overlay[:-3]**2)
        ax1.plot(x_extrap, m1_no_aa * x_extrap, 'C0--', alpha=0.3, label=None)

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
    ax1.set_xlim(1.5, None)
    
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
    fig_no_aa, ax1 = plt.subplots(1, 1, figsize=(90 / 25.4, 70 / 25.4))

    x_no_aa = df_no_aa['Power (mE)'].values
    x_extrap = np.linspace(min(2, x_no_aa.min()), max(2, x_no_aa.max()), 100)

    # Plot K3 (no AA)
    k3_no_aa = get_numeric(df_no_aa['K3 (s⁻¹)'])
    k3_plot = ax1.errorbar(x_no_aa, k3_no_aa,
                yerr=get_error(df_no_aa['K3 (s⁻¹)']), fmt='^', label=r'$k_3$',
                linewidth=1, markersize=4, alpha=1, color='C1', capsize=3)
    # Linear fit for K3 with zero intercept
    m3 = np.sum(x_no_aa[:] * k3_no_aa[:]) / np.sum(x_no_aa[:]**2)
    k3_extrap = m3 * 2
    print(m3)
    ax1.plot(x_extrap, m3 * x_extrap, 'C1--', label=None)

    # Plot K4 (no AA)
    k4_no_aa = get_numeric(df_no_aa['K4 (s⁻¹)'])
    k4_plot = ax1.errorbar(x_no_aa,  k4_no_aa,
                yerr=get_error(df_no_aa['K4 (s⁻¹)']), fmt='v', label=r'$k_4$',
                linewidth=1, markersize=4, alpha=1, color='C2', capsize=3)
    # Linear fit for K4 with zero intercept
    m4 = np.sum(x_no_aa[:] * k4_no_aa[:]) / np.sum(x_no_aa[:]**2)
    k4_extrap = m4
    ax1.plot(x_extrap, m4 * x_extrap, 'C2--', label=None)

    # Plot K1 (no AA)
    k1_no_aa = get_numeric(df_no_aa['K1 (s⁻¹)'])
    k1_plot = ax1.errorbar(x_no_aa, k1_no_aa,
                 yerr=get_error(df_no_aa['K1 (s⁻¹)']), fmt='o', label=r'$k_1$',
                 linewidth=1, markersize=4, alpha=1, color='C0', capsize=3)
    # Linear fit for K1 with zero intercept
    m1 = np.sum(x_no_aa[:-3] * k1_no_aa[:-3]) / np.sum(x_no_aa[:-3]**2)
    k1_extrap = m1 * 2
    ax1.plot(x_extrap, m1 * x_extrap, 'C0--', label=None)

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

    ax1.set_ylabel(r'Kinetic rate (s$^{-1}$)')
    ax1.set_xlabel(r'Photon flux density (mmol photons m$^{-2}$ s$^{-1}$)')

    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax1.set_yscale('log')
    ax1.set_xlim(10, None)
    
    # Custom legend without error bars
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', label=r'$k_1$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='s', color='C3', label=r'$k_2$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='^', color='C1', label=r'$k_3$', linestyle='None', markersize=4),
        Line2D([0], [0], marker='v', color='C2', label=r'$k_4$', linestyle='None', markersize=4)
    ]
    ax1.legend(handles=legend_elements, loc='best', frameon=False)

    sns.despine()
    plt.tight_layout()
    plot_file_no_aa = os.path.join(base_data_dir, 'k_values_vs_power_no_AA.png')
    plt.savefig(plot_file_no_aa, dpi=300, bbox_inches='tight')
    print(f"No AA plot saved to: {plot_file_no_aa}")
    plt.show()
else:
    print("No data to plot")
