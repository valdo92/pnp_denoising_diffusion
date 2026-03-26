import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for better-looking plots
sns.set_theme(style="whitegrid")

import ast # add this import at the top of the file

def load_metrics(exp_name, base_dir="results"):
    """
    Loads the metrics CSV file for a given experiment and perfectly parses PyTorch tensor strings.
    """
    folder_name = f"results_{exp_name}"
    csv_name = f"metrics_{exp_name}.csv"
    file_path = os.path.join(base_dir, folder_name, csv_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return None
        
    df = pd.read_csv(file_path)
    
    # Force robust cleaning on all columns except the image name
    for col in df.columns:
        if "image_name" in col:
            continue
            
        def clean_val(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val)
            
            # If the CSV wrote "tensor(45.67, device='cuda:0')", we clear the tensor wrapper
            if "tensor" in val_str:
                val_str = val_str.replace("tensor(", "").split(",")[0].replace(")", "").strip()
            
            # If it's a list string like "[45.67]" 
            if val_str.startswith("["):
                val_str = val_str.strip("[]")
                
            try:
                return float(val_str)
            except ValueError:
                return np.nan # If it completely fails, return NaN to avoid crashing the mean()

        # Apply the cleaner row by row and force float64
        df[col] = df[col].apply(clean_val).astype('float64')
            
    return df

def analyze_exp1():
    print("--- Analyzing Experiment 1: Gamma Ablation (PGD vs HQS) ---")
    
    # 1. Gather Data
    baseline_df = load_metrics("EXP1_HQS_baseline")
    
    gammas = [1.0, 10.0, 20.0, 50.0]
    pgd_dfs = {}
    for g in gammas:
        df = load_metrics(f"EXP1_PGD_gamma_{int(g)}")
        if df is not None:
            pgd_dfs[g] = df
            
    if baseline_df is None or not pgd_dfs:
        print("Required data for Experiment 1 is missing. Aborting plot.")
        return

    # 2. Compute Averages
    # We take the mean across all 100 images for each metric
    # PSNR known is removed from the plot but can stay in the calculation if needed.
    # We remove it from metrics_to_plot so it's totally ignored.
    metrics_to_plot = ['boundary_tv', 'lpips']    
    results = {
        'Method': [],
        'Gamma': []
    }
    for m in metrics_to_plot:
        results[m] = []
        # Add standard deviation for error bars
        results[f"{m}_std"] = []

    # Process HQS
    results['Method'].append('HQS')
    results['Gamma'].append(np.nan) # HQS doesn't use the PGD gamma
    for m in metrics_to_plot:
        results[m].append(baseline_df[m].mean())
        results[f"{m}_std"].append(baseline_df[m].std())

    # Process PGDs
    for g, df in pgd_dfs.items():
        results['Method'].append('PGD')
        results['Gamma'].append(g)
        for m in metrics_to_plot:
            results[m].append(df[m].mean())
            results[f"{m}_std"].append(df[m].std())

    # Note: Global metrics like FID are usually stored as single values or NaNs in per-image rows
    # We extract the first valid FID we find.
    has_fid = 'Global_FID' in baseline_df.columns
    if has_fid:
        results['Global_FID'] = [baseline_df['Global_FID'].dropna().iloc[0]]
        for g, df in pgd_dfs.items():
            results['Global_FID'].append(df['Global_FID'].dropna().iloc[0])

    res_df = pd.DataFrame(results)
    print("\nSummarized Results:")
    print(res_df.to_string(index=False))

    # 3. Plotting (Modifié pour ne faire que 2 colonnes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Experiment 1: PGD Gamma Ablation vs HQS Baseline', fontsize=16)

    # Prepare x-axis labels. Treat HQS as a categorical point.
    x_labels = ['HQS\nBaseline'] + [f'PGD\n$\gamma={int(g)}$' for g in gammas]
    x_pos = np.arange(len(x_labels))

    # Metric 1: Boundary TV (Seam Smoothness) - Lower is better
    ax = axes[0]
    ax.errorbar(x_pos, res_df['boundary_tv'], yerr=res_df['boundary_tv_std'], fmt='-o', color='green', capsize=5, capthick=2, markersize=8)
    ax.scatter(0, res_df['boundary_tv'].iloc[0], color='red', s=100, zorder=5, label='HQS Analytic Limit')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Boundary Total Variation')
    ax.set_title('Seam Smoothness (Lower TV is smoother)')
    ax.legend()

    # Metric 2: Global FID or LPIPS
    ax = axes[1]
    if has_fid:
        ax.plot(x_pos, res_df['Global_FID'], '-o', color='purple', markersize=8)
        ax.scatter(0, res_df['Global_FID'].iloc[0], color='red', s=100, zorder=5, label='HQS Analytic Limit')
        ax.set_ylabel('Global FID (~100 samples)')
        ax.set_title('Overall Generation Quality (FID)')
    else:
        ax.errorbar(x_pos, res_df['lpips'], yerr=res_df['lpips_std'], fmt='-o', color='purple', capsize=5)
        ax.scatter(0, res_df['lpips'].iloc[0], color='red', s=100, zorder=5, label='HQS Analytic Limit')
        ax.set_ylabel('LPIPS Distance')
        ax.set_title('Perceptual Distance')
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig('experiment1_metrics_plot.png', dpi=300)
    print("\nSaved plot to 'experiment1_metrics_plot.png'")
    try:
        plt.show()
    except Exception as e:
        print("Note: Plot displayed or interrupted:", e)

if __name__ == "__main__":
    analyze_exp1()