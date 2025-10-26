import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from tkinter.filedialog import askopenfilenames

# Function to load event data 
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

# Select multiple files
event_files = askopenfilenames(title="Select Event Files")

rows = 260
cols = 346
window_size = 10
output_dir = Path("plots/consistency")
output_dir.mkdir(parents=True, exist_ok=True)

for event_file in event_files:
    print(f"\nProcessing {event_file} ...")
    events = load_events(event_file)

    # Extract frequency from filename
    match = re.search(r'(\d+)Hz', event_file)
    frequency = int(match.group(1)) if match else 80

    #  Separate ON and OFF events
    on_events = events[events[:, 3] == 1]
    off_events = events[events[:, 3] == -1]

    # Accumulate events into heatmap for ON events
    heatmap = np.zeros((rows, cols))
    for t, x, y, pol in on_events:
        heatmap[int(y), int(x)] += 1  # y = row, x = col

    # Sliding window to find best ROI
    max_sum = 0
    best_window = (0, 0)
    for y in range(rows - window_size + 1):
        for x in range(cols - window_size + 1):
            roi_sum = np.sum(heatmap[y:y+window_size, x:x+window_size])
            if roi_sum > max_sum:
                max_sum = roi_sum
                best_window = (x, y)

    x_bottom = best_window[0]
    x_top = x_bottom + window_size
    y_bottom = best_window[1]
    y_top = y_bottom + window_size
    print(f"Best ROI window: x={x_bottom}-{x_top-1}, y={y_bottom}-{y_top-1}, total ON events={max_sum}")

    # Function to compute ROI temporal consistency (std per pixel)
    # The purpose is to see how regularly each pixel fires within the ROI
    # If a pixel fires very regularly, it has low std (more consistent)
    # If a pixel fires irregularly, it has high std (less consistent)
    # Pixel firing rate = number of events / total time span
    # Use std of inter-event intervals as a measure of consistency - lower is better because it means events are more evenly spaced in time
    # This implies a more stable response to the stimulus > less noise

    def compute_roi_consistency(events_subset):
        pixel_times = defaultdict(list)
        for t, x, y, pol in events_subset:
            if x_bottom <= x < x_top and y_bottom <= y < y_top:
                pixel_times[(int(x), int(y))].append(t)

        roi_consistency = np.full((window_size, window_size), np.nan)
        for (x, y), times in pixel_times.items():
            times = np.array(times)
            if len(times) > 1:
                roi_consistency[y - y_bottom, x - x_bottom] = np.std(np.diff(times))
        return roi_consistency

    roi_consistency_on = compute_roi_consistency(on_events)
    roi_consistency_off = compute_roi_consistency(off_events)

    # Save .npy arrays for precise analysis
    np.save(output_dir / f"{Path(event_file).stem}_roi_consistency_ON.npy", roi_consistency_on)
    np.save(output_dir / f"{Path(event_file).stem}_roi_consistency_OFF.npy", roi_consistency_off)


    # Darker pixels = more consistent (lower std)
    # Lighter pixels = less consistent (higher std)
    # Plot temporal consistency of ROI (ON events only)
    plt.figure(figsize=(6, 6))
    plt.imshow(roi_consistency_on, cmap='plasma', origin='upper',
               extent=[x_bottom, x_top, y_top, y_bottom])
    plt.colorbar(label='std(inter-event interval)')
    plt.xlabel("X pixels")
    plt.ylabel("Y pixels")
    plt.title(f"Temporal Consistency (ROI) — {frequency} Hz (ON)")
    plt.tight_layout()

    # Save figure
    save_path = output_dir / f"{Path(event_file).stem}_roi_consistency.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved temporal consistency map to: {save_path}")
    print(f"Saved .npy arrays for ON/OFF ROI: {output_dir / Path(event_file).stem}_roi_consistency_*.npy")