from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilenames
import re

# Parameters
nbins = 36
window_size = 10
rows = 260
cols = 346
output_dir = Path("plots/spatial_noise")
output_dir.mkdir(parents=True, exist_ok=True)


# Load events from CSV files
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

# Find ROI window
def find_roi_window(on_events, rows, cols, window_size):
    """Sliding window to find ROI with highest ON activity"""
    heatmap = np.zeros((rows, cols), dtype=int)
    for t, x, y, p in on_events:
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


# Compute per-pixel counts in ON/OFF phases
def counts_per_pixel_in_phases(events, freq_hz, nbins=36):
    if events.size==0:
        return {}, {}, {}, {}
    times = events[:,0].astype(float)
    times -= times.min()
    period = 1.0/float(freq_hz)
    phases = (times % period) / period  # in [0,1)
    bins = np.linspace(0,1,nbins+1)
    bin_inds = np.digitize(phases, bins) - 1
    bin_inds = np.clip(bin_inds,0,nbins-1)

    on_bins = set(range(0, nbins//2))
    off_bins = set(range(nbins//2, nbins))

    on_on = defaultdict(int)
    on_off = defaultdict(int)
    off_on = defaultdict(int)
    off_off = defaultdict(int)

    for i, b in enumerate(bin_inds):
        x = int(events[i,1]); y = int(events[i,2]); p = int(events[i,3])
        key = (x,y)
        if p == 1:  # ON polarity
            if b in on_bins:
                on_on[key] += 1
            else:
                on_off[key] += 1
        else:       # OFF polarity
            if b in on_bins:
                off_on[key] += 1
            else:
                off_off[key] += 1

    return on_on, on_off, off_on, off_off

# Compute per-pixel counts and SNR using ROI
def compute_count_snr_per_file(stim_file, baseline_file, freq_hz):
    stim = load_events(stim_file)
    base = load_events(baseline_file)

    # compute per-pixel counts in phases
    on_on, on_off, off_on, off_off = counts_per_pixel_in_phases(stim, freq_hz, nbins=nbins)

    # baseline counts per pixel
    base_on = defaultdict(int)
    base_off = defaultdict(int)
    for t,x,y,p in base:
        key=(int(x),int(y))
        if int(p)==1:
            base_on[key]+=1
        else:
            base_off[key]+=1

    # all pixels
    keys = set()
    keys.update(on_on.keys(), on_off.keys(), off_on.keys(), off_off.keys(), base_on.keys(), base_off.keys())

    rows_list=[]
    for k in sorted(keys):
        sON = on_on.get(k,0)
        sOFF = off_off.get(k,0)
        nON = base_on.get(k,0)
        nOFF = base_off.get(k,0)
        rows_list.append({
            "x": k[0], "y": k[1],
            "sON": int(sON), "nON": int(nON),
            "sOFF": int(sOFF), "nOFF": int(nOFF)
        })

    df = pd.DataFrame(rows_list)
    return df

# Compute ROI SNR using only pixels in roi_pixels
def compute_roi_snr(df_pix, roi_pixels):
    roi_df = df_pix[df_pix[['x','y']].apply(tuple, axis=1).isin(roi_pixels)]
    
    sON_total = roi_df["sON"].sum()
    nON_total = roi_df["nON"].sum()
    sOFF_total = roi_df["sOFF"].sum()
    nOFF_total = roi_df["nOFF"].sum()

    # derived from Moeys et al.
    roi_snr_on_db = 20.0 * np.log10((sON_total - nON_total)/nON_total) if (nON_total>0 and sON_total>nON_total) else -np.inf
    roi_snr_off_db = 20.0 * np.log10((sOFF_total - nOFF_total)/nOFF_total) if (nOFF_total>0 and sOFF_total>nOFF_total) else -np.inf

    return roi_snr_on_db, roi_snr_off_db


# Run and choose files for analysis
stim_files = askopenfilenames(title="Select stimulation files (AC)")
baseline_file = "recordings/baseline.csv"

all_summaries = []

for f in stim_files:
    match = re.search(r'(\d+)Hz', f)
    freq = int(match.group(1)) if match else 80

    df_pix = compute_count_snr_per_file(f, baseline_file, freq)

    # find ROI based on ON events in stim
    on_events = load_events(f)
    roi_top_left, _ = find_roi_window([e for e in on_events if e[3]==1], rows, cols, window_size)
    roi_pixels = [(x, y) for x in range(roi_top_left[0], roi_top_left[0]+window_size)
                          for y in range(roi_top_left[1], roi_top_left[1]+window_size)]

    roi_snr_on, roi_snr_off = compute_roi_snr(df_pix, roi_pixels)

    base = Path(f).stem
    df_pix.to_csv(output_dir / f"{base}_snr_counts.csv", index=False)

    all_summaries.append({
        "file": base, "freq": freq,
        "roi_snr_on_db": roi_snr_on, "roi_snr_off_db": roi_snr_off
    })

pd.DataFrame(all_summaries).to_csv(output_dir / "roi_snr_counts_summary.csv", index=False)
print("Done.")
