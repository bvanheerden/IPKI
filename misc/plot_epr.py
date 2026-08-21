import sys
import os

# Add the project root to sys.path to allow imports of utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import utils

utils.setup_plotting()

# Determine the path to EPR.csv on Desktop
desktop_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
default_path = os.path.join(desktop_dir, 'EPR.csv')
if not os.path.exists(default_path):
    default_path = r'C:\Users\bertu\Desktop\EPR.csv'


def load_epr_data(file_path=default_path):
    """
    Loads EPR data from a CSV file.
    Supports both European format (semicolon delimited, comma decimal)
    and standard CSV formats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EPR data file not found at: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            sep=';',
            decimal=',',
            header=None,
            names=['Magnetic Field (G)', 'EPR Signal (a.u.)']
        )
        if df.shape[1] == 2 and np.issubdtype(df.dtypes.iloc[0], np.number) and np.issubdtype(df.dtypes.iloc[1], np.number):
            return df
    except Exception:
        pass

    return pd.read_csv(file_path)


def plot_epr(df=None, file_path=default_path, show=True, save_path=None):
    """
    Plots EPR data using matplotlib.
    """
    if df is None:
        df = load_epr_data(file_path)

    field = df.iloc[:, 0]
    signal = df.iloc[:, 1]

    fig, ax = plt.subplots(figsize=utils.get_figure_size(50, 40))
    ax.plot(field, signal, lw=1.5, color='k')
    ax.set_xlabel(str(df.columns[0]) if df.columns[0] else 'Magnetic Field (G)')
    ax.set_ylabel(str(df.columns[1]) if df.columns[1] else 'Intensity (a.u.)')
    ax.set_xlim(3200, 3500)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    data = load_epr_data()
    plot_epr(data)
