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
spatial_noise_threshold = 0.5   # pixels with noise > threshold are flagged as abnormal
output_dir = Path("plots/spatial_noise")
output_dir.mkdir(parents=True, exist_ok=True)

# Load events
def load_events(file_path):
    """Load CSV-like event data: t,x,y,pol"""
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
    events_np = events_np[events_np[:, 0].argsort()]  # sort by time
    return events_np

# Find ROI Window for ON events
def find_roi_window(on_events, rows, cols, window_size):
    """Sliding window to find ROI with highest ON activity"""
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


# Compute per-pixel spatial noise within ROI and flag abnormal pixels
def compute_spatial_noise(events, freq_hz, threshold=spatial_noise_threshold):
    roi_events, phases, pols, xs, ys, roi_bounds = compute_phase_info(events, freq_hz)
    if roi_events is None:
        return None, roi_bounds, None

    inds = np.digitize(phases, np.linspace(0,1,nbins+1)) - 1
    inds = np.clip(inds, 0, nbins-1)

    # ROI-level histograms for expected polarity
    roi_hist_on = np.zeros(nbins, dtype=int)
    roi_hist_off = np.zeros(nbins, dtype=int)
    for i, b in enumerate(inds):
        if pols[i] == 1:
            roi_hist_on[b] += 1
        else:
            roi_hist_off[b] += 1
    expected_pol = np.where(roi_hist_on >= roi_hist_off, 1, -1)

    # per-pixel histograms
    pixel_on = defaultdict(lambda: np.zeros(nbins, dtype=int))
    pixel_off = defaultdict(lambda: np.zeros(nbins, dtype=int))
    pixel_total = defaultdict(int)

    for i, b in enumerate(inds):
        x, y, p = int(xs[i]), int(ys[i]), int(pols[i])
        pixel_total[(x,y)] += 1
        if p == 1:
            pixel_on[(x,y)][b] += 1
        else:
            pixel_off[(x,y)][b] += 1

    # compute spatial noise score and flag
    spatial_noise = np.full((window_size, window_size), np.nan)
    abnormal_mask = np.zeros((window_size, window_size), dtype=bool)
    x_bottom, x_top, y_bottom, y_top = roi_bounds

    for (x,y), tot in pixel_total.items():
        if tot < min_events_per_pixel:
            continue
        on_hist = pixel_on[(x,y)]
        off_hist = pixel_off[(x,y)]
        matches = np.sum(on_hist[expected_pol==1]) + np.sum(off_hist[expected_pol==-1])
        noise_score = 1 - matches / (tot + 1)
        spatial_noise[y - y_bottom, x - x_bottom] = noise_score
        if noise_score >= threshold:
            abnormal_mask[y - y_bottom, x - x_bottom] = True

    return spatial_noise, roi_bounds, abnormal_mask


# Run and choose files to analyze
event_files = askopenfilenames(title="Select Event Files")
spatial_noise_maps = []
frequencies = []

for f in event_files:
    events = load_events(f)
    match = re.search(r'(\d+)Hz', f)
    freq_hz = int(match.group(1)) if match else 80

    spatial_noise_map, roi_bounds, abnormal_mask = compute_spatial_noise(events, freq_hz)
    spatial_noise_maps.append(spatial_noise_map)
    frequencies.append(freq_hz)

    if spatial_noise_map is None:
        print(f"No ROI events for {f}, skipping.")
        continue

    # Save CSV per pixel
    x_bottom, x_top, y_bottom, y_top = roi_bounds
    pixels = []
    for yy in range(window_size):
        for xx in range(window_size):
            score = spatial_noise_map[yy, xx]
            abnormal = abnormal_mask[yy, xx]
            pixels.append({
                "x": x_bottom+xx,
                "y": y_bottom+yy,
                "SpatialNoiseScore": score,
                "Abnormal": abnormal
            })
    df_pixels = pd.DataFrame(pixels)
    csv_path = output_dir / f"{Path(f).stem}_spatial_noise.csv"
    df_pixels.to_csv(csv_path, index=False)

    # Plot heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(spatial_noise_map, origin='lower', cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Spatial Noise Score')
    plt.title(f"Spatial Noise — {Path(f).stem}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{Path(f).stem}_spatial_noise.png")
    plt.close()
    print(f"Saved spatial noise map and CSV for {f}")

# Compute final mean and standard deviation
summary = []

for f, freq, noise_map in zip(event_files, frequencies, spatial_noise_maps):
    if noise_map is None:
        continue
    scores = noise_map.flatten()
    scores = scores[~np.isnan(scores)]
    mean_noise = np.mean(scores)
    std_noise = np.std(scores)
    
    summary.append({
        "Filename": Path(f).name,
        "Frequency_Hz": freq,
        "Mean_Spatial_Noise": mean_noise,
        "Std_Spatial_Noise": std_noise
    })

df_summary = pd.DataFrame(summary).sort_values("Frequency_Hz")
summary_csv = output_dir / "spatial_noise_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"Saved spatial noise summary CSV to {summary_csv}")

# Plotting
plt.figure(figsize=(8,5))
plt.errorbar(df_summary["Frequency_Hz"], df_summary["Mean_Spatial_Noise"],
             yerr=df_summary["Std_Spatial_Noise"], fmt='o-', capsize=5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Spatial Noise ± Std")
plt.title("Spatial Noise per Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "spatial_noise_mean_std.png")
plt.show()