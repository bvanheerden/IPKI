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


def plot_dataset_rates_linear(df, dataset_name, fit_slices=None, inset_max_power=None, save_path=None):
    """
    Plot a single dataset on a separate figure with 3 subplots (k1, k3, k4) on linear axes.

    Parameters:
    - df: DataFrame containing 'Power (mE)', 'K1 (s⁻¹)', 'K3 (s⁻¹)', 'K4 (s⁻¹)'
    - dataset_name: Title/label of the dataset
    - fit_slices: Optional dict mapping rate name ('k1', 'k3', 'k4') to slice/indices for linear fitting
    - inset_max_power: Optional maximum power (mE) to display in zoom-in insets (e.g. 72 mE for thylakoids)
    - save_path: Optional file path to save the generated figure
    """
    if df.empty:
        print(f"Skipping {dataset_name}: dataset is empty.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 58 / 25.4), sharex=True)

    x = df['Power (mE)'].values.astype(float)
    x_max = np.max(x)
    x_fit = np.linspace(0, x_max * 1.05, 100)

    rate_specs = [
        ('K1 (s⁻¹)', r'$k_1$', 'C0', 'o', 'k1', 0),
        ('K3 (s⁻¹)', r'$k_3$', 'C1', '^', 'k3', 1),
        ('K4 (s⁻¹)', r'$k_4$', 'C2', 'v', 'k4', 2),
    ]

    print(f"\n==================== {dataset_name} (Linear Fits) ====================")

    for col_name, rate_symbol, color, marker, key, idx in rate_specs:
        ax = axes[idx]
        y = get_numeric(df[col_name]).values
        y_err = get_error(df[col_name]).values

        # Plot data points with error bars on linear scale
        ax.errorbar(
            x, y, yerr=y_err, fmt=marker, color=color,
            linewidth=1, markersize=4, capsize=3, label=f'Data {rate_symbol}'
        )

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

        # Plot linear fit line and confidence interval
        ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=1.5, label=f'Fit {rate_symbol}')
        ax.fill_between(x_fit, y_low, y_high, color=color, alpha=0.2)

        # Inset for low-power region (e.g. 72 mE down)
        if inset_max_power is not None:
            mask_inset = x <= (inset_max_power + 0.5)
            x_ins = x[mask_inset]
            y_ins = y[mask_inset]
            y_err_ins = y_err[mask_inset]

            # Inset axes positioned in the upper-left of the subplot
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
        # ax.set_title(rate_symbol, fontsize=8)
        ax.set_xlabel(r'Photon flux density (mmol m$^{-2}$ s$^{-1}$)')
        ax.set_ylabel(f'{rate_symbol} ' + r'(s$^{-1}$)')
        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # fig.suptitle(dataset_name, fontsize=9, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")

    return fig


def main():
    base_data_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = load_datasets()

    # Fit subsets matching linear ranges from power studies if desired
    # (can set to None to fit all available points)
    fit_subsets = {
        'LHCII': {
            'k1': slice(0, -2),  # Excludes saturated highest power points
            'k3': slice(0, -2),
            'k4': slice(0, -2)
        },
        'Thylakoids +AA': {
            'k1': slice(0, 5),   # Linear regime up to 72 mmol m^-2 s^-1
            'k3': slice(0, 5),
            'k4': slice(0, 7)
        },
        'Thylakoids -AA': {
            'k1': slice(0, 5),
            'k3': slice(0, 3),   # Linear regime up to 72 mmol m^-2 s^-1
            'k4': slice(0, 3)
        }
    }

    # Cutoff for zoom-in insets (72 mE for thylakoid plots)
    inset_cutoffs = {
        'Thylakoids +AA': 72.0,
        'Thylakoids -AA': 72.0,
        'LHCII': None
    }

    file_slugs = {
        'LHCII': 'k_values_vs_power_LHCII_linear.png',
        'Thylakoids +AA': 'k_values_vs_power_with_AA_linear.png',
        'Thylakoids -AA': 'k_values_vs_power_no_AA_linear.png'
    }

    figs = []
    for dataset_name, df in datasets.items():
        save_path = os.path.join(base_data_dir, file_slugs.get(dataset_name, f"{dataset_name}_linear.png"))
        fig = plot_dataset_rates_linear(
            df=df,
            dataset_name=dataset_name,
            fit_slices=fit_subsets.get(dataset_name),
            inset_max_power=inset_cutoffs.get(dataset_name),
            save_path=save_path
        )
        if fig is not None:
            figs.append(fig)

    plt.show()


if __name__ == '__main__':
    main()
