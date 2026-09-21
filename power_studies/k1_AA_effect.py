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
        path_no_aa = os.path.join(results_dir, 'data_with_aa_1q_lowk2.pkl')
        if os.path.exists(path_no_aa):
            with open(path_no_aa, 'rb') as f:
                df_no_aa = pickle.load(f)
                df_no_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
        else:
            print("data_no_aa_1q.pkl not found in results/")

        path_with_aa = os.path.join(results_dir, 'data_with_aa_1q_highk2.pkl')
        if os.path.exists(path_with_aa):
            with open(path_with_aa, 'rb') as f:
                df_with_aa = pickle.load(f)
                df_with_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
        else:
            print("data_with_aa_1q.pkl not found in results/")

    path_lhcii = os.path.join(results_dir, 'data_with_aa_1q_midk2.pkl')
    if os.path.exists(path_lhcii):
        with open(path_lhcii, 'rb') as f:
            df_lhcii_no_aa = pickle.load(f)
            df_lhcii_no_aa.sort_values(by='Power (mE)', inplace=True, ignore_index=True)
    else:
        print("data_no_aa_lhcii.pkl not found in results/")

    return {
        'mid k2': df_lhcii_no_aa,
        'low k2': df_no_aa,
        'high k2': df_with_aa,
    }


def plot_dataset_rates_linear(df, dataset_name, axes=None, fit_slices=None,
                              inset_max_power=None, save_path=None):
    """
    Plot a single dataset across 4 subplots (k1, k2, k3, k4) on linear axes.

    Parameters:
    - df: DataFrame containing 'Power (mE)', 'K1 (s⁻¹)', 'K3 (s⁻¹)', 'K4 (s⁻¹)' (and optionally 'K2 (s⁻¹)')
    - dataset_name: Title/label of the dataset
    - axes: Optional array-like of 4 Axes objects to plot into. If None, a new figure is created.
    - fit_slices: Optional dict mapping rate name ('k1', 'k2', 'k3', 'k4') to slice/indices for linear fitting
    - inset_max_power: Optional maximum power (mE) to display in zoom-in insets (e.g. 72 mE for thylakoids)
    - save_path: Optional file path to save the generated figure (when axes is None)
    """
    if df.empty:
        print(f"Skipping {dataset_name}: dataset is empty.")
        return None

    fig = axes.figure

    if dataset_name == 'high k2':
        rate_specs = [
            ('K1 (s⁻¹)', r'$k_2 = 1.5$', 'C1', 'o', 'k1', 0),
        ]
    elif dataset_name == 'low k2':
        rate_specs = [
            ('K1 (s⁻¹)', r'$k_2 = 0.7$', 'C2', 's', 'k1', 0),
        ]
    else:
        rate_specs = [
            ('K1 (s⁻¹)', r'$k_2 = 1$', 'C0', '^', 'k1', 0),
        ]


    x = df['Power (mE)'].values.astype(float)
    x_max = np.max(x)
    x_fit = np.linspace(0, x_max * 1.05, 100)

    print(f"\n==================== {dataset_name} (Linear Fits) ====================")

    for col_name, rate_symbol, color, marker, key, idx in rate_specs:
        ax = axes
        y = get_numeric(df[col_name]).values
        y_err = get_error(df[col_name]).values

        # Determine points to include in linear fit
        if fit_slices and key in fit_slices and fit_slices[key] is not None:
            sl = fit_slices[key]
            x_for_fit = x[sl]
            y_for_fit = y[sl]
        else:
            x_for_fit = x
            y_for_fit = y

        # Linear fit and confidence band
        m, b, cov = linear_fit(x_for_fit, y_for_fit)
        y_fit = m * x_fit + b
        y_low, y_high = get_fit_bounds(x_fit, m, b, cov)

        slope_err = np.sqrt(cov[0, 0]) if cov[0, 0] >= 0 else 0.0
        intercept_err = np.sqrt(cov[1, 1]) if cov[1, 1] >= 0 else 0.0
        print(f"  {rate_symbol}: slope = {m:.3e} ± {slope_err:.3e}, intercept = {b:.3e} ± {intercept_err:.3e} (fitted {len(x_for_fit)}/{len(x)} points)")

        if key not in ['k2', 'k2a', 'k2b']:
            # Plot data points with error bars on linear scale
            ax.errorbar(
                x, y, yerr=y_err, fmt=marker, color=color,
                linewidth=1, markersize=4, capsize=3, label=rate_symbol
            )
            # Plot linear fit line and confidence interval
            # ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=1.5, label=f'Fit {rate_symbol}')
            # ax.fill_between(x_fit, y_low, y_high, color=color, alpha=0.2)
            # ax.set_ylim(bottom=0)


        # Configure linear axes
        is_first_col = ax.get_subplotspec().is_first_col() if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None else (idx == 0)
        is_last_row = ax.get_subplotspec().is_last_row() if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None else True

        # ax.set_xlim(0, x_max * 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    return fig


def main():
    base_data_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets()

    # Fit subsets matching linear ranges from power studies if desired
    # (can set to None to fit all available points)
    fit_subsets = {
        'high k2': {
            'k1': slice(0, 5),   # Linear regime up to 72 mmol m^-2 s^-1
        },
        'low k2': {
            'k1': slice(0, 5),
        }
    }

    fig, ax = plt.subplots(1, 1, figsize=(90 / 25.4, 55 / 25.4))

    for col_idx, (dataset_name, df) in enumerate(datasets.items()):
        print(dataset_name)
        plot_dataset_rates_linear(
            df=df,
            dataset_name=dataset_name,
            axes=ax,
            fit_slices=fit_subsets.get(dataset_name),
            inset_max_power=None
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel(r'Photon flux density (mmol m$^{-2}$ s$^{-1}$)')
    ax.set_ylabel(r'$k_1$ (s$^{-1}$)')

    plt.tight_layout()
    save_path = os.path.join(base_data_dir, 'k_values_vs_power_all_linear.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined 4x3 grid plot to: {save_path}")

    plt.show()


if __name__ == '__main__':
    main()
