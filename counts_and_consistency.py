# The following code is meant as a summary to consolidate results from event_counts.py and event_consistency.py.
# It assumes both scripts have been run and produced their respective output files.
        # Loads event_count_summary.csv from event_counts.py.
        # Loads ROI temporal consistency arrays (.npy) produced by event_consistency.py.

# Computes:
        # Average temporal variability (ROI_avg_std) per ROI.
        # Fraction of events outside the ROI (Outside_ROI_ratio).
        # Compare ON vs OFF event consistency.

# Final Output:
        # Flags ROIs as noisy or reliable based on statistical thresholds (tunable)
        # Saves a combined summary CSV.
        # Prints a quick view of noisy vs reliable ROIs.

import pandas as pd
import numpy as np
from pathlib import Path

# Configure paths
csv_path = Path("plots/event_count_summary.csv")  # Output from event_counts.py
roi_folder = Path("plots/consistency")            # Folder containing _roi_consistency_ON.npy and _roi_consistency_OFF.npy
save_dir = roi_folder
save_dir.mkdir(exist_ok=True, parents=True)

# Threshold parameters (analysis done with both 1 and 2 std above mean)
n_std_above_mean = 1  # classify as Noisy if ROI_avg_std > mean + n_std_above_mean * std

# Load event counts CSV
df_counts = pd.read_csv(csv_path)
df_counts["Frequency_Hz"] = pd.to_numeric(df_counts["Frequency_Hz"], errors="coerce")
df_counts = df_counts.dropna(subset=["Frequency_Hz"]).sort_values("Frequency_Hz")

# Initialize lists to store average std values
roi_avg_std_on_list = []
roi_avg_std_off_list = []

# Compute average std of inter-event intervals within ROI for ON and OFF events
for idx, row in df_counts.iterrows():
    base_name = Path(row['Filename']).stem
    
    # Load ON ROI array
    on_file = roi_folder / f"{base_name}_roi_consistency_ON.npy"
    if on_file.exists():
        roi_on = np.load(on_file)
        finite_vals = roi_on[np.isfinite(roi_on)]
        roi_avg_std_on = finite_vals.mean() if len(finite_vals) > 0 else np.nan
    else:
        roi_avg_std_on = np.nan

    # Load OFF ROI array
    off_file = roi_folder / f"{base_name}_roi_consistency_OFF.npy"
    if off_file.exists():
        roi_off = np.load(off_file)
        finite_vals = roi_off[np.isfinite(roi_off)]
        roi_avg_std_off = finite_vals.mean() if len(finite_vals) > 0 else np.nan
    else:
        roi_avg_std_off = np.nan

    roi_avg_std_on_list.append(roi_avg_std_on)
    roi_avg_std_off_list.append(roi_avg_std_off)

df_counts["ROI_avg_std_ON"] = roi_avg_std_on_list
df_counts["ROI_avg_std_OFF"] = roi_avg_std_off_list

# Classify ROIs based on average std thresholds as Noisy or Reliable
# ON
mean_on = df_counts["ROI_avg_std_ON"].mean()
std_on = df_counts["ROI_avg_std_ON"].std()
threshold_on = mean_on + n_std_above_mean * std_on
df_counts["ROI_Classification_ON"] = df_counts["ROI_avg_std_ON"].apply(
    lambda x: "Noisy" if x > threshold_on else "Reliable"
)

# OFF
mean_off = df_counts["ROI_avg_std_OFF"].mean()
std_off = df_counts["ROI_avg_std_OFF"].std()
threshold_off = mean_off + n_std_above_mean * std_off
df_counts["ROI_Classification_OFF"] = df_counts["ROI_avg_std_OFF"].apply(
    lambda x: "Noisy" if x > threshold_off else "Reliable"
)

# Save combined summary CSV
out_csv = save_dir / "combined_event_count_consistency_summary_ON_OFF.csv"
df_counts.to_csv(out_csv, index=False)

# Print summary of classifications
summary_cols = [
    "Filename", "Frequency_Hz",
    "ROI_Events_ON", "ROI_Events_OFF",
    "ROI_Ratio_ON", "ROI_Ratio_OFF",
    "ROI_avg_std_ON", "ROI_Classification_ON",
    "ROI_avg_std_OFF", "ROI_Classification_OFF"
]

print("\n=== Combined Event Count & Temporal Consistency Summary (ON/OFF) ===")
print(df_counts[summary_cols].to_string(index=False))
print(f"\n✅ Saved combined summary to: {out_csv}")
