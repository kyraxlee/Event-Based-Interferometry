import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilenames

# Choose multiple files for analysis
event_files = askopenfilenames(title="Select Event Files for all frequencies")

# Load events from CSV
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
    events_np = events_np[events_np[:, 0].argsort()]  # Sort by time
    return events_np

# Define ROI parameters 
x_lower, x_upper = 0, 346
y_lower, y_upper = 0, 260
window_size = 10
timestep = 1000  # microseconds

all_temporal_counts = []  # store temporal counts for all files
frequency_labels = []

for fpath in event_files:
    events = load_events(fpath)
    
    # Filter ON events in main area
    focused_events = np.array([
        e for e in events 
        if x_lower <= e[1] < x_upper and y_lower <= e[2] < y_upper and e[3] == 1
    ])
    focused_events[:, 1] -= x_lower
    focused_events[:, 2] -= y_lower

    # Create heatmap to find ROI
    rows = y_upper - y_lower
    cols = x_upper - x_lower
    heatmap = np.zeros((rows, cols))
    for t, x, y, pol in focused_events:
        x, y = int(x), int(y)
        heatmap[y, x] += 1

    # Find ROI with max activity
    max_sum = 0
    best_window = (0, 0)
    for y in range(rows - window_size + 1):
        for x in range(cols - window_size + 1):
            roi_sum = np.sum(heatmap[y:y+window_size, x:x+window_size])
            if roi_sum > max_sum:
                max_sum = roi_sum
                best_window = (x, y)
    x_bottom, y_bottom = best_window
    x_top = x_bottom + window_size - 1
    y_top = y_bottom + window_size - 1

    # Filter events in ROI
    roi_events = np.array([
        e for e in focused_events
        if x_bottom <= e[1] <= x_top and y_bottom <= e[2] <= y_top
    ])
    if roi_events.size == 0:
        all_temporal_counts.append(np.array([0]))
        frequency_labels.append(Path(fpath).stem)
        continue
    
    # Temporal binning
    roi_events[:, 0] -= roi_events[:, 0].min()
    n_bins = int(np.ceil(roi_events[:, 0].max() / timestep))
    temporal_counts = np.zeros(n_bins)
    for t, x, y, pol in roi_events:
        bin_idx = int(t // timestep)
        temporal_counts[bin_idx] += 1
    
    all_temporal_counts.append(temporal_counts)
    frequency_labels.append(Path(fpath).stem)

# Create boxplot across all frequencies
plt.figure(figsize=(12, 6))
box = plt.boxplot(all_temporal_counts, labels=frequency_labels, vert=True, patch_artist=True)

# Overlay mean ± std inside each box
for i, counts in enumerate(all_temporal_counts, 1):
    m = counts.mean()
    s = counts.std()
    plt.text(i, m + s, f"{m:.1f}±{s:.1f}", ha='center', va='bottom', fontsize=9)

plt.ylabel("Number of events per bin")
plt.title("Temporal Event Distributions Across Frequencies (ROI)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
