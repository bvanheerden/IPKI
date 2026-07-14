import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
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

def trast_model(tau, tau_T, A, tau_D, A_D, tau_bl, model_type='triplet_dark'):
    """
    TRAST model for triplet state + another reversible dark state + bleaching.
    """
    triplet = A * (1 - (1 - np.exp(-tau / tau_T)) / (tau / tau_T))
    bleaching = (1 - np.exp(-tau / tau_bl)) / (tau / tau_bl)

    if model_type == 'one_dark':
        return (1 - triplet) * bleaching
    
    dark_state2 = A_D * (1 - (1 - np.exp(-tau / tau_D)) / (tau / tau_D))
    return (1 - triplet - dark_state2) * bleaching

def test_models():
    tau = np.logspace(-6, 0, 100)
    
    # Model 1: Triplet + Dark + Bleaching
    y1 = trast_model(tau, 1e-6, 0.2, 10e-3, 0.1, 2, model_type='triplet_dark')
    
    # Model 2: One Dark + Bleaching
    y2 = trast_model(tau, 10e-3, 0.2, 0, 0, 2, model_type='one_dark')
    
    plt.figure()
    plt.semilogx(tau, y1, label='Triplet + Dark')
    plt.semilogx(tau, y2, label='One Dark')
    plt.legend()
    plt.savefig('test_models.png')
    print("Test plot saved as test_models.png")

if __name__ == "__main__":
    test_models()