# The goal is to try and find the noise events that don’t match the expected signal modulation cycle. 
# Where we have OFF events where ON should be and vice versa.

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilenames
import re

# Parameters
nbins = 36
min_events_per_pixel = 5
window_size = 10
rows = 260
cols = 346
output_dir = Path("plots/temporal_noise")
output_dir.mkdir(parents=True, exist_ok=True)

# Load Events
def load_events(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Events file not found: {file_path}")
    data = []
    with p.open('r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('t'):
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            t, x, y, pol = parts
            t = float(t)
            x = int(x)
            y = int(y)
            pol = -1 if int(pol) == 0 else 1
            data.append((t, x, y, pol))
    events_np = np.array(data, dtype=np.float64)
    events_np = events_np[events_np[:,0].argsort()]
    return events_np

# Find ROI Window
def find_roi_window(on_events, rows, cols, window_size):
    heatmap = np.zeros((rows, cols), dtype=int)
    for t,x,y,p in on_events:
        heatmap[int(y), int(x)] += 1
    max_sum = 0
    best = (0,0)
    for yy in range(rows - window_size + 1):
        for xx in range(cols - window_size + 1):
            s = heatmap[yy:yy+window_size, xx:xx+window_size].sum()
            if s > max_sum:
                max_sum = s
                best = (xx, yy)
    return best, max_sum

# Compute ROI and fractional phase for events
def compute_phase_info(events, freq_hz):
    on_mask = events[:,3] == 1
    on_events = events[on_mask]
    (x_bottom, y_bottom), _ = find_roi_window(on_events, rows, cols, window_size)
    x_top = x_bottom + window_size
    y_top = y_bottom + window_size
    in_roi_mask = (events[:,1] >= x_bottom) & (events[:,1] < x_top) & (events[:,2] >= y_bottom) & (events[:,2] < y_top)
    roi_events = events[in_roi_mask]
    if roi_events.shape[0] == 0:
        return None, None, None, None, None, (x_bottom, x_top, y_bottom, y_top)
    timestamps = roi_events[:,0].astype(float)
    timestamps -= timestamps.min()
    pols = roi_events[:,3].astype(int)
    xs = roi_events[:,1].astype(int)
    ys = roi_events[:,2].astype(int)
    period = 1.0 / freq_hz
    phases = (timestamps % period) / period
    return roi_events, phases, pols, xs, ys, (x_bottom, x_top, y_bottom, y_top)

# Metric 1: Weighted mismatch computes the ratio of all phase-polarity inversions to total ROI events
# Metric 2: Normalized pixel mismatch computes the mean per-pixel mismatch fraction within the ROI, normalized by
# active pixel count

def weighted_mismatch_fraction(events, freq_hz):
    roi_events, phases, pols, xs, ys, roi_bounds = compute_phase_info(events, freq_hz)
    if roi_events is None:
        return np.nan, roi_bounds
    roi_hist_on = np.zeros(nbins, dtype=int)
    roi_hist_off = np.zeros(nbins, dtype=int)
    inds = np.digitize(phases, np.linspace(0,1,nbins+1)) - 1
    inds = np.clip(inds, 0, nbins-1)
    for i, b in enumerate(inds):
        if pols[i] == 1:
            roi_hist_on[b] += 1
        else:
            roi_hist_off[b] += 1
    expected_pol = np.where(roi_hist_on >= roi_hist_off, 1, -1)
    total_mismatches = np.sum([pols[i] != expected_pol[inds[i]] for i in range(len(inds))])
    roi_mismatch_fraction = total_mismatches / len(roi_events)
    return roi_mismatch_fraction, roi_bounds

def normalized_pixel_mismatch(events, freq_hz):
    roi_events, phases, pols, xs, ys, roi_bounds = compute_phase_info(events, freq_hz)
    if roi_events is None:
        return np.nan, roi_bounds
    inds = np.digitize(phases, np.linspace(0,1,nbins+1)) - 1
    inds = np.clip(inds, 0, nbins-1)
    pixel_on = defaultdict(lambda: np.zeros(nbins, dtype=int))
    pixel_off = defaultdict(lambda: np.zeros(nbins, dtype=int))
    pixel_total = defaultdict(int)
    roi_hist_on = np.zeros(nbins, dtype=int)
    roi_hist_off = np.zeros(nbins, dtype=int)
    for i, b in enumerate(inds):
        x, y, p = xs[i], ys[i], pols[i]
        pixel_total[(x,y)] += 1
        if p == 1:
            pixel_on[(x,y)][b] += 1
            roi_hist_on[b] += 1
        else:
            pixel_off[(x,y)][b] += 1
            roi_hist_off[b] += 1
    expected_pol = np.where(roi_hist_on >= roi_hist_off, 1, -1)
    per_pixel_mismatch_rates = []
    for (x,y), tot in pixel_total.items():
        if tot < min_events_per_pixel:
            continue
        on_hist = pixel_on[(x,y)]
        off_hist = pixel_off[(x,y)]
        mism = np.sum(on_hist[expected_pol==-1]) + np.sum(off_hist[expected_pol==1])
        matches = np.sum(on_hist[expected_pol==1]) + np.sum(off_hist[expected_pol==-1])
        denom = matches + mism
        per_pixel_mismatch_rates.append(mism/denom if denom>0 else np.nan)
    normalized_roi_mismatch = np.nanmean(per_pixel_mismatch_rates) if per_pixel_mismatch_rates else np.nan
    return normalized_roi_mismatch, roi_bounds

# Run and choose files for analysis
event_files = askopenfilenames(title="Select Event Files")
results = []

for f in event_files:
    events = load_events(f)
    match = re.search(r'(\d+)Hz', f)
    freq_hz = int(match.group(1)) if match else 80

    weighted_frac, _ = weighted_mismatch_fraction(events, freq_hz)
    norm_pixel, _ = normalized_pixel_mismatch(events, freq_hz)

    results.append({
        "Filename": Path(f).name,
        "Frequency_Hz": freq_hz,
        "Weighted_Mismatch_Fraction": weighted_frac,
        "Normalized_Pixel_Mismatch": norm_pixel
    })

df_results = pd.DataFrame(results).sort_values("Frequency_Hz")
csv_out = output_dir / "temporal_noise_summary.csv"
df_results.to_csv(csv_out, index=False)
print(f"Saved summary CSV to {csv_out}")

# Plotting
plt.figure(figsize=(8,5))
plt.plot(df_results["Frequency_Hz"], df_results["Weighted_Mismatch_Fraction"], 'o-', label="Weighted Mismatch Fraction")
plt.plot(df_results["Frequency_Hz"], df_results["Normalized_Pixel_Mismatch"], 's-', label="Normalized Pixel Mismatch")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mismatch Rate")
plt.title("Temporal Noise Across Frequencies")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "temporal_noise_comparison.png")
plt.show()
