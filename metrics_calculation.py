import os
import pandas as pd
import numpy as np

def clean_val(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    if "tensor" in val_str:
        val_str = val_str.replace("tensor(", "").split(",")[0].replace(")", "").strip()
    if val_str.startswith("["):
        val_str = val_str.strip("[]")
    try:
        return float(val_str)
    except ValueError:
        return np.nan

# Define paths
hqs_path = "results/results_EXP3_HQS_fast_iter20/metrics_EXP3_HQS_fast_iter20.csv"
pgd_path = "results/results_EXP3_PGD_fast_iter20_gamma_20/metrics_EXP3_PGD_fast_iter20_gamma_20.csv"

if os.path.exists(hqs_path) and os.path.exists(pgd_path):
    df_hqs = pd.read_csv(hqs_path)
    df_pgd = pd.read_csv(pgd_path)
    
    print("=== HQS (NFE=20) ===")
    hqs_tv = df_hqs['boundary_tv'].apply(clean_val)
    hqs_lpips = df_hqs['lpips'].apply(clean_val)
    print(f"Boundary TV : {hqs_tv.mean():.2f} ± {hqs_tv.std():.2f}")
    print(f"LPIPS       : {hqs_lpips.mean():.3f} ± {hqs_lpips.std():.3f}")
    
    print("\n=== PGD (NFE=20, Gamma=20) ===")
    pgd_tv = df_pgd['boundary_tv'].apply(clean_val)
    pgd_lpips = df_pgd['lpips'].apply(clean_val)
    print(f"Boundary TV : {pgd_tv.mean():.2f} ± {pgd_tv.std():.2f}")
    print(f"LPIPS       : {pgd_lpips.mean():.3f} ± {pgd_lpips.std():.3f}")
else:
    print("Files not found. Check the paths.")