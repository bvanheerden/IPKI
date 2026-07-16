import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
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

if __name__ == "__main__":
    base_path = "blinking/SMS_blinking"
    summary = calculate_switching_rates(base_path)
    
    print("\nSwitching Rate Summary per Condition:")
    print("-" * 75)
    print(f"{'Condition':<25} | {'Avg Rate (s^-1)':<15} | {'Std Error':<12} | {'Particles':<10}")
    print("-" * 75)
    
    def sort_key(cond):
        match = re.match(r"(\d+\.?\d*)%", cond)
        if match:
            return (0, float(match.group(1)))
        return (1, cond)

    for condition, data in sorted(summary.items(), key=lambda x: sort_key(x[0])):
        print(f"{condition:<25} | {data['avg_rate']:<15.4f} | {data['std_err']:<12.4f} | {data['count']:<10}")

    value = summary.pop('Compressed air')
    summary['Air'] = value
    value = summary.pop('GOC+N2 purging')
    summary['GOC+N$_2$'] = value
    value = summary.pop('N2 purging')
    summary['N$_2$'] = value
    value = summary.pop('1.2%')
    summary['1.2% O$_2$'] = value
    value = summary.pop('4.2%')
    summary['4.2% O$_2$'] = value
    value = summary.pop('10.2%')
    summary['10.2% O$_2$'] = value

    # Plotting
    conditions_sorted = sorted(summary.keys(), key=sort_key)
    avg_rates = [summary[c]['avg_rate'] for c in conditions_sorted]
    std_errs = [summary[c]['std_err'] for c in conditions_sorted]
    
    plt.figure(figsize=(90/25.4, 40/25.4))
    bars = plt.bar(conditions_sorted, avg_rates, yerr=std_errs, capsize=3, color='C0', edgecolor='black', lw=0.7,
                   error_kw=dict(elinewidth=1))
    plt.xlabel('Condition')
    plt.ylabel('Switching rate (s$^{-1}$)')
    plt.xticks(fontsize=5)
    
    # Add counts on top of bars
    # for bar, c in zip(bars, conditions_sorted):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height + std_errs[conditions_sorted.index(c)],
    #              f'n={summary[c]["count"]}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_plot = "switching_rates_bar_plot.png"
    plt.savefig(output_plot, dpi=300)
    print(f"\nBar plot saved to {output_plot}")
    plt.show()
