import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
import utils

utils.setup_plotting()

def calculate_switching_rates(base_path):
    results = {}
    
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist.")
        return
    
    # Iterate over conditions (directories in SMS_blinking)
    conditions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for condition in sorted(conditions):
        condition_path = os.path.join(base_path, condition)
        rates = []
        
        # Walk through subfolders
        for root, dirs, files in os.walk(condition_path):
            for file in files:
                # Match the required file naming patterns
                if file.endswith(" levels (ROI).csv") or file.endswith(" levels-grouped (ROI).csv"):
                    # Avoid "-plot" files if they were accidentally matched (though endswith should handle it)
                    if "-plot" in file:
                        continue
                        
                    file_path = os.path.join(root, file)
                    try:
                        # Load the CSV
                        df = pd.read_csv(file_path)
                        
                        if df.empty:
                            continue
                            
                        n_levels = len(df)
                        
                        # Manual overrides for specific particles
                        if "4.2%" in condition and "LHCII@PLL-385nW-2020-01-30-Sample00" in root and "Particle 72" in file:
                            n_levels = 2
                        elif "Compressed air" in condition and "LHCII@PVA-Air-385nW-Sample00" in root:
                            if "Particle 1 " in file or "Particle 1." in file: # particle 1
                                n_levels = 2
                            elif "Particle 13" in file:
                                n_levels = 2
                        elif "GOC" in condition and "LH2-0.4_PVA@2000nW#01" in root:
                            if "Particle 3 " in file or "Particle 3." in file:
                                n_levels = 3
                        elif "N2+GOC" in condition:
                            if "SMS-LH2-0.4_PVA@2000nW-Sample1" in root:
                                if any(p in file for p in ["Particle 5 ", "Particle 5.", "Particle 7 ", "Particle 7.", "Particle 11 ", "Particle 11."]):
                                    n_levels = 3
                                elif "Particle 19 " in file or "Particle 19." in file:
                                    n_levels = 17
                            elif "SMS-LH2-0.4_PVA@2000nW-Sample3" in root:
                                if "Particle 2 " in file or "Particle 2." in file:
                                    n_levels = 13
                                elif any(p in file for p in ["Particle 9 ", "Particle 9.", "Particle 38 ", "Particle 38.", "Particle 39 ", "Particle 39.", "Particle 40 ", "Particle 40.", "Particle 68 ", "Particle 68.", "Particle 73 ", "Particle 73.", "Particle 78 ", "Particle 78.", "Particle 119 ", "Particle 119."]):
                                    n_levels = 3
                                elif any(p in file for p in ["Particle 66 ", "Particle 66.", "Particle 108 ", "Particle 108."]):
                                    n_levels = 5
                                elif "Particle 104 " in file or "Particle 104." in file:
                                    n_levels = 15

                        # We need at least one level to calculate rate (though rate with 1 level is 0)
                        if n_levels < 1:
                            continue
                            
                        # Get the "End Time (s)" of the last level
                        # Check column names (handle potential variations in casing/spacing)
                        col_name = None
                        for col in df.columns:
                            if col.strip().lower() == "end time (s)":
                                col_name = col
                                break
                        
                        if col_name is None:
                            print(f"Warning: 'End Time (s)' column not found in {file_path}")
                            continue
                            
                        last_end_time = df[col_name].iloc[-1]
                        
                        if last_end_time > 0:
                            switching_rate = (n_levels - 1) / last_end_time
                            rates.append(switching_rate)
                        else:
                            print(f"Warning: Last end time is 0 in {file_path}")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        if rates:
            avg_rate = np.mean(rates)
            std_err = np.std(rates) / np.sqrt(len(rates)) if len(rates) > 1 else 0.0
            results[condition] = {
                'avg_rate': avg_rate,
                'std_err': std_err,
                'count': len(rates),
                'rates': rates
            }
        else:
            results[condition] = {
                'avg_rate': 0.0,
                'std_err': 0.0,
                'count': 0,
                'rates': []
            }
            
    return results

def plot_switching_rates(summary, dataset_name, output_plot):
    if not summary:
        print(f"No data to plot for {dataset_name}")
        return

    # Renaming and sorting for SMS_blinking
    if dataset_name == "SMS_blinking":
        # Rename conditions for better display
        renames = {
            'Compressed air': 'Air',
            'GOC+N2 purging': 'GOC+N$_2$',
            'N2 purging': 'N$_2$',
            '1.2%': '1.2% O$_2$',
            '4.2%': '4.2% O$_2$',
            '10.2%': '10.2% O$_2$'
        }
        for old, new in renames.items():
            if old in summary:
                summary[new] = summary.pop(old)

        def sort_key(cond):
            match = re.match(r"(\d+\.?\d*)%", cond)
            if match:
                return (0, float(match.group(1)))
            return (1, cond)
        
        sorted_keys = sorted(summary.keys(), key=sort_key)
        # However, the original code sorted by height at the end.
        # Let's keep the user's "sort by height" request for the final plot.
    elif dataset_name == "LH2-O2-data":
        renames = {
            'Ambient air': 'Air',
            'N2+GOC': 'N$_2$+GOC'
        }
        for old, new in renames.items():
            if old in summary:
                summary[new] = summary.pop(old)
    
    # Sort conditions by average rate (descending order)
    conditions_sorted = sorted(summary.keys(), key=lambda c: summary[c]['avg_rate'], reverse=True)
    avg_rates = [summary[c]['avg_rate'] for c in conditions_sorted]
    std_errs = [summary[c]['std_err'] for c in conditions_sorted]

    if dataset_name == "SMS_blinking":
        plt.figure(figsize=(120/25.4, 40/25.4))
    else:
        plt.figure(figsize=(50/25.4, 40/25.4))
    
    # Color logic
    if dataset_name == "SMS_blinking":
        colors = ['C7' if c == 'Air' else 'C7' for c in conditions_sorted]
        legend_elements = [Rectangle((0, 0), 0, 0, color='C6', label='PVA'),
                           Rectangle((0, 0), 0, 0, color='C7', label='Buffer'),]
    elif dataset_name == "LH2-O2-data":
        colors = ['C8' if 'Air' in c else 'C8' for c in conditions_sorted]
        legend_elements = [Rectangle((0, 0), 0, 0, color='C0', label='PVA (LH2)'),
                           Rectangle((0, 0), 0, 0, color='C1', label='Buffer (LH2)'),]
    else:
        colors = 'C0' # Default color for other datasets
        legend_elements = None

    bars = plt.bar(conditions_sorted, avg_rates, yerr=std_errs, capsize=3, color=colors, edgecolor='black', lw=0.7,
                   error_kw=dict(elinewidth=1))
    
    plt.xlabel('Condition')
    plt.ylabel('Switching frequency (s$^{-1}$)')
    plt.xticks(fontsize=5)
    
    # Add values on top of bars
    for bar, c in zip(bars, conditions_sorted):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + summary[c]['std_err'] + 0.03,
                 f'{summary[c]["avg_rate"]:.2g}', ha='center', va='bottom', fontsize=5, color='black')

    # if legend_elements:
    #     plt.legend(handles=legend_elements, frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"\nBar plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    datasets = [
        ("blinking/SMS_blinking", "SMS_blinking", "switching_rates_bar_plot.png"),
        ("blinking/LH2-O2-data", "LH2-O2-data", "switching_rates_LH2_O2_bar_plot.png")
    ]
    
    for path, name, plot_name in datasets:
        print(f"\n{'='*20} Processing {name} {'='*20}")
        summary = calculate_switching_rates(path)
        
        print("\nSwitching Rate Summary per Condition:")
        print("-" * 75)
        print(f"{'Condition':<25} | {'Avg Rate (s^-1)':<15} | {'Std Error':<12} | {'Particles':<10}")
        print("-" * 75)
        
        # Internal print sorting (lexicographical/numeric if possible)
        def print_sort_key(cond):
            match = re.match(r"(\d+\.?\d*)%", cond)
            if match:
                return (0, float(match.group(1)))
            return (1, cond)

        for condition, data in sorted(summary.items(), key=lambda x: print_sort_key(x[0])):
            print(f"{condition:<25} | {data['avg_rate']:<15.4f} | {data['std_err']:<12.4f} | {data['count']:<10}")
            
        plot_switching_rates(summary, name, plot_name)
