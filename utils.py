import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def setup_plotting():
    """Sets up global plotting parameters for consistency."""
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
    sns.set_palette('deep')

def load_spectrum(filename, datadir, existing_df, run, skiprows=36):
    """
    Loads spectral data from a CSV/ASC file.
    
    Args:
        filename: Name of the file.
        datadir: Directory containing the file.
        existing_df: DataFrame to append the new data to.
        run: Label for the current run/time.
        skiprows: Number of rows to skip in the CSV.
        
    Returns:
        Updated DataFrame.
    """
    filepath = os.path.join(datadir, filename)
    new_data = pd.read_csv(
        filepath, 
        names=['Wavelength (nm)', 'Intensity (counts)'], 
        usecols=[0, 1], 
        dtype=np.float64,
        skiprows=skiprows
    )
    new_data['run'] = run
    return pd.concat([existing_df, new_data], ignore_index=True)

def get_figure_size(width_mm, height_mm):
    """Converts mm to inches for matplotlib figsize."""
    return (width_mm / 25.4, height_mm / 25.4)
