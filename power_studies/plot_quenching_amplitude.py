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
    """Extract numeric rate values from series (handles 'value ± err' format)."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or series.dtype in ['O', 'object', 'string', object]:
        return series.astype(str).str.split(' ±').str[0].astype(float)
    return series.astype(float)


def get_error(series):
    """Extract error values from series (handles 'value ± err' format)."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or series.dtype in ['O', 'object', 'string', object]:
        parts = series.astype(str).str.split(' ±')
        return parts.apply(lambda x: float(x[1]) if len(x) > 1 else 0.0)
    return pd.Series(0.0, index=series.index)


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

    # Fallback if only Kr1 is present
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


def compute_quenching_amplitude(df):
    """
    Compute quenching amplitude A = k1 / (k1 + k2) and propagated uncertainty.
    """
    k1 = get_numeric(df['K1 (s⁻¹)']).values
    k1_err = get_error(df['K1 (s⁻¹)']).values
    k2, k2_err = get_k2_average(df)
    k2 = k2.values
    k2_err = k2_err.values

    denom = k1 + k2
    amplitude = np.where(denom != 0, k1 / denom, 0.0)

    # Error propagation: dA/dk1 = k2 / (k1 + k2)^2, dA/dk2 = -k1 / (k1 + k2)^2
    denom_sq = denom**2
    amplitude_err = np.where(
        denom_sq != 0,
        np.sqrt((k2 * k1_err)**2 + (k1 * k2_err)**2) / denom_sq,
        0.0
    )

    return amplitude, amplitude_err


def linear_fit(x, y, weights=None):
    """
    Fit a linear model (y = m * x + b or y = m * x) to data.
    Returns slope m, intercept b, and covariance matrix.
    """
    if FIT_WITH_INTERCEPT:
        p, cov = np.polyfit(x, y, 1, w=weights, cov=True)
        return p[0], p[1], cov
    else:
        # Manual calculation for zero-intercept fit: y = m * x
        x_sq_sum = np.sum(x**2)
        m = np.sum(x * y) / x_sq_sum
        resid = y - m * x
        n = len(x)
        sigma2 = np.sum(resid**2) / (n - 1) if n > 1 else 0
        m_var = sigma2 / x_sq_sum
        cov = np.array([[m_var, 0], [0, 0]])
        return m, 0.0, cov


def get_fit_bounds(x_range, m, b, cov):
    """Calculate 1-sigma uncertainty bounds for linear fit."""
    y_fit = m * x_range + b
    var_y = (x_range**2 * cov[0, 0]) + (2 * x_range * cov[0, 1]) + cov[1, 1]
    sig_y = np.sqrt(np.maximum(var_y, 0))
    return y_fit - sig_y, y_fit + sig_y


def load_datasets(results_dir=None):
    """
    Load the 3 datasets: LHCII, Thylakoids +AA, and Thylakoids -AA.
    """
    if results_dir is None:
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
        if 'Power (mE)' in df_no_aa.columns:
            df_no_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
        if 'Power (mE)' in df_with_aa.columns:
            df_with_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
    else:
        path_no_aa = os.path.join(results_dir, 'data_no_aa_1q.pkl')
        if os.path.exists(path_no_aa):
            with open(path_no_aa, 'rb') as f:
                df_no_aa = pickle.load(f)
                df_no_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
        else:
            print("data_no_aa_1q.pkl not found in results/")

        path_with_aa = os.path.join(results_dir, 'data_with_aa_1q.pkl')
        if os.path.exists(path_with_aa):
            with open(path_with_aa, 'rb') as f:
                df_with_aa = pickle.load(f)
                df_with_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
        else:
            print("data_with_aa_1q.pkl not found in results/")

    path_lhcii = os.path.join(results_dir, 'data_no_aa_lhcii.pkl')
    if os.path.exists(path_lhcii):
        with open(path_lhcii, 'rb') as f:
            df_lhcii_no_aa = pickle.load(f)
            df_lhcii_no_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
    else:
        print("data_no_aa_lhcii.pkl not found in results/")

    return {
        'LHCII': df_lhcii_no_aa,
        'Thylakoids +AA': df_with_aa,
        'Thylakoids -AA': df_no_aa
    }


def plot_dataset_quenching_amplitude(df, dataset_name, ax=None, fit_slice=None, inset_max_power=None):
    """
    Plot quenching amplitude k1/(k1+k2) as a function of power for a single dataset on given axis ax.

    Parameters:
    - df: DataFrame containing power and rate columns
    - dataset_name: Title/label of the dataset
    - ax: Matplotlib axis to plot on
    - fit_slice: Optional slice/indices for linear fitting range
    - inset_max_power: Optional maximum power (mE) to display in zoom-in insets (e.g. 72 mE for thylakoids)
    """
    if df.empty:
        print(f"Skipping {dataset_name}: dataset is empty.")
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(85 / 25.4, 65 / 25.4))
        standalone = True
    else:
        standalone = False

    x = df['Power (mE)'].values.astype(float)
    x_max = np.max(x)
    x_fit = np.linspace(0, x_max * 1.05, 100)

    y, y_err = compute_quenching_amplitude(df)

    color = 'C0'
    marker = 'o'

    print(f"\n==================== {dataset_name} (Quenching Amplitude) ====================")

    # Plot data points with error bars on linear scale
    ax.errorbar(
        x, y, yerr=y_err, fmt=marker, color=color,
        linewidth=1, markersize=4, capsize=3, label=r'Data $k_1/(k_1+k_2)$'
    )

    # Determine points to include in linear fit
    if fit_slice is not None:
        x_for_fit = x[fit_slice]
        y_for_fit = y[fit_slice]
    else:
        x_for_fit = x
        y_for_fit = y

    # Linear fit and confidence band
    m, b, cov = linear_fit(x_for_fit, y_for_fit)
    y_fit = m * x_fit + b
    y_low, y_high = get_fit_bounds(x_fit, m, b, cov)

    slope_err = np.sqrt(cov[0, 0]) if cov[0, 0] >= 0 else 0.0
    intercept_err = np.sqrt(cov[1, 1]) if cov[1, 1] >= 0 else 0.0
    print(f"  Quenching Amplitude: slope = {m:.3e} ± {slope_err:.3e}, intercept = {b:.3e} ± {intercept_err:.3e} (fitted {len(x_for_fit)}/{len(x)} points)")

    # Plot linear fit line and confidence interval
    ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=1.5, label='Fit')
    ax.fill_between(x_fit, y_low, y_high, color=color, alpha=0.2)

    # Inset for low-power region (e.g. 72 mE down)
    if inset_max_power is not None:
        mask_inset = x <= (inset_max_power + 0.5)
        x_ins = x[mask_inset]
        y_ins = y[mask_inset]
        y_err_ins = y_err[mask_inset]

        # Inset axes positioned in the upper-left of the plot
        ax_ins = ax.inset_axes([0.18, 0.53, 0.44, 0.41])
        ax_ins.patch.set_facecolor('white')
        ax_ins.patch.set_alpha(1.0)

        # Inset data points
        ax_ins.errorbar(
            x_ins, y_ins, yerr=y_err_ins, fmt=marker, color=color,
            linewidth=0.8, markersize=3, capsize=2
        )

        # Inset fit curve and bounds
        x_fit_ins_max = inset_max_power * 1.1
        x_fit_ins = np.linspace(0, x_fit_ins_max, 100)
        y_fit_ins = m * x_fit_ins + b
        y_low_ins, y_high_ins = get_fit_bounds(x_fit_ins, m, b, cov)

        ax_ins.plot(x_fit_ins, y_fit_ins, color=color, linestyle='-', linewidth=1.2)
        ax_ins.fill_between(x_fit_ins, y_low_ins, y_high_ins, color=color, alpha=0.2)

        ax_ins.set_xlim(0, x_fit_ins_max)
        ax_ins.set_ylim(bottom=0)
        ax_ins.tick_params(axis='both', which='major', labelsize=6, pad=1.5, length=2.5)
        ax_ins.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
        ax_ins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))

        # Add zoom indicator box and connecting lines
        indicator = ax.indicate_inset_zoom(ax_ins, edgecolor='black', alpha=0.5)
        if hasattr(indicator, 'connectors') and indicator.connectors is not None:
            indicator.connectors[2].set_visible(True)   # Lower-right connector
            indicator.connectors[3].set_visible(False)  # Upper-right connector
        elif isinstance(indicator, (tuple, list)) and len(indicator) == 2:
            indicator[1][2].set_visible(True)
            indicator[1][3].set_visible(False)

        ax.legend(frameon=False, loc='lower right', fontsize=6.5)
    else:
        ax.legend(frameon=False, loc='best')

    # Configure linear axes
    ax.set_title(dataset_name, fontsize=8)
    ax.set_xlabel(r'Photon flux density (mmol m$^{-2}$ s$^{-1}$)')
    ax.set_ylabel(r'Quenching amplitude $k_1 / (k_1 + k_2)$')
    ax.set_xlim(0, x_max * 1.05)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    if standalone:
        plt.tight_layout()
        plt.show()


def main():
    base_data_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets()

    # Fit subsets matching linear ranges from power studies if desired
    # (can set to None to fit all available points)
    fit_subsets = {
        'LHCII': slice(0, -2),          # Excludes saturated highest power points
        'Thylakoids +AA': slice(0, 5),   # Linear regime up to 72 mmol m^-2 s^-1
        'Thylakoids -AA': slice(0, 5)    # Linear regime up to 72 mmol m^-2 s^-1
    }

    # Cutoff for zoom-in insets (72 mE for thylakoid plots)
    inset_cutoffs = {
        'Thylakoids +AA': 72.0,
        'Thylakoids -AA': None,
        'LHCII': None
    }

    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 65 / 25.4), sharey=False)

    for ax, (dataset_name, df) in zip(axes, datasets.items()):
        plot_dataset_quenching_amplitude(
            df=df,
            dataset_name=dataset_name,
            ax=ax,
            fit_slice=fit_subsets.get(dataset_name),
            inset_max_power=inset_cutoffs.get(dataset_name)
        )

    plt.tight_layout()
    save_path = os.path.join(base_data_dir, 'quenching_amplitude_vs_power.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
