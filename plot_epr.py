import sys
import os

# Add the project root to sys.path to allow imports of utils
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from misc.plot_epr import load_epr_data, plot_epr, default_path

if __name__ == '__main__':
    data = load_epr_data(default_path)
    plot_epr(data)
