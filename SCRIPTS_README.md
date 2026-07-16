# Project Organization and Script Documentation

This project has been reorganized to improve maintainability and reduce code duplication. Common functions for plotting and data loading have been centralized in `utils.py` at the project root.

## Project Structure

- `spectra/`: Scripts for loading and plotting spectral data (DHE, SOSG).
- `power_studies/`: Scripts for analyzing and plotting power-dependent blinking data.
- `trast/`: Scripts related to TRansient Absorption Spectroscopy (TRAST).
- `kinetic_models/`: Core kinetic models and scripts for fitting them to experimental data.
- `misc/`: Miscellaneous utility scripts and experiments.
- `blinking/`: Experimental data directory.
- `notebooks/`: Jupyter notebooks for exploratory analysis and visualization.
- `results/`: Processed data files (.pkl), plots (.png), and CSV results.

## Centralized Utilities (`utils.py`)

- `setup_plotting()`: Configures Matplotlib rcParams for consistent, publication-quality figures.
- `load_spectrum(filename, datadir, existing_df, run, skiprows)`: A robust function for loading spectral CSV/ASC data.
- `get_figure_size(width_mm, height_mm)`: Converts millimeter dimensions to inches for figure creation.

## Script Descriptions

### Spectra (`spectra/`)
- `DHE_spec.py`: Loads and plots DHE (Dihydroethidium) fluorescence spectra under different conditions (with/without LHCII and SOD).
- `SOSG Spec.py`: Analyzes Singlet Oxygen Sensor Green (SOSG) spectra to track singlet oxygen production.
- `SOD_bar_plot.py`: Generates bar plots for SOD-related experimental results.

### Power Studies (`power_studies/`)
- `Power_studies_June2026.py`: Main script for processing power study datasets from June 2026.
- `global_power_fit.py`: Performs global fitting across multiple power levels.
- `local_power_fit.py`: Fits individual power levels separately.
- `mixed_effects_power_fit.py`: Uses mixed-effects models for power study analysis.
- `plot_power_studies.py`: Visualization script for power study results.

### TRAST (`trast/`)
- `TRAST.py`: Basic TRAST data processing.
- `hdf_TRAST.py`: TRAST analysis specifically for HDF5 data formats.
- `hdf_TRAST_triplet_dark.py`: TRAST analysis focused on triplet state dynamics in the dark.

### Kinetic Models (`kinetic_models/`)
- `kinetic_model.py`: The core mathematical model (ODE system) and trace averaging functions.
- `fit_k2_exponential.py`: Fits exponential decays to the $k_2$ kinetic parameter.
- `test_k2_light.py`: Investigates the effect of light on the $k_2$ parameter.
- `test_model_option.py`: Utility for testing different model configurations.

### Miscellaneous (`misc/`)
- `AA_trace_compare.py`: Compares fluorescence traces with and without Ascorbic Acid (AA).
- `IPKI_example.py`: Example script for IPKI (Iterative Partial Knowledge Integration) analysis.
- `chemicals.py`: Configuration and metadata for different chemical treatments.
- `control_sod_only.py`: Control experiments for SOD treatments.
- `inspect_pkl.py`: Utility for inspecting the contents of `.pkl` data files.

## How to Run
All scripts in subdirectories have been configured to correctly import `utils.py` and `kinetic_model.py` from their respective locations. You can run them directly from their folders:
```bash
python power_studies/plot_power_studies.py
```
