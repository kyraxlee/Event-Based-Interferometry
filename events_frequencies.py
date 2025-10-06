import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import re

# -------------------------------
# 1. Select event files
# -------------------------------
root = tk.Tk()
root.withdraw()
event_files = askopenfilenames(title="Select Event Files")
if not event_files:
    raise ValueError("No event files selected.")

# -------------------------------
# 2. Load event data
# -------------------------------
def load_events(file_path):
    p = Path(file_path)
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
            data.append((float(t), int(x), int(y), int(pol)))
    events = np.array(data, dtype=np.float64)
    if events.size == 0:
        return events
    events = events[events[:, 0].argsort()]
    events[:, 0] -= events[:, 0].min()  # normalize time start
    return events

# -------------------------------
# 3. Parameters
# -------------------------------
x_lower, x_upper = 0, 346
y_lower, y_upper = 0, 260
window_size = 10

# -------------------------------
# 4. Process each file
# -------------------------------
frequencies = []
event_counts = []

for file_path in event_files:
    events = load_events(file_path)
    if events.size == 0:
        continue

    # Filter ON events
    focused = np.array([
        e for e in events
        if x_lower <= e[1] < x_upper and y_lower <= e[2] < y_upper and e[3] == 1
    ])
    if focused.size == 0:
        continue

    rows = y_upper - y_lower
    cols = x_upper - x_lower

    # Build heatmap
    heatmap = np.zeros((rows, cols))
    for t, x, y, pol in focused:
        heatmap[int(y - y_lower), int(x - x_lower)] += 1

    # Find most active ROI
    max_sum, best_window = 0, (0, 0)
    for y in range(rows - window_size + 1):
        for x in range(cols - window_size + 1):
            roi_sum = np.sum(heatmap[y:y+window_size, x:x+window_size])
            if roi_sum > max_sum:
                max_sum = roi_sum
                best_window = (x, y)

    x0, y0 = best_window
    x1, y1 = x0 + window_size, y0 + window_size

    # Count events in ROI
    roi_events = [
        e for e in focused
        if x0 <= (e[1] - x_lower) < x1 and y0 <= (e[2] - y_lower) < y1
    ]
    count = len(roi_events)

    # Extract frequency from filename (e.g. "85Hz.csv")
    fname = Path(file_path).stem
    match = re.search(r'(\d+)\s*Hz', fname, re.IGNORECASE)
    freq = float(match.group(1)) if match else np.nan

    frequencies.append(freq)
    event_counts.append(count)

    print(f"{fname}: {count} ON events in ROI ({x0},{y0})")

# -------------------------------
# 5. Dot plot
# -------------------------------
freqs = np.array(frequencies)
counts = np.array(event_counts)

# Sort by frequency
order = np.argsort(freqs)
freqs = freqs[order]
counts = counts[order]

# Scale to "dots per thousand events"
dots_per_thousand = 1  # 1 dot = 100 events
dots = (counts / 100 * dots_per_thousand).astype(int)

plt.figure(figsize=(10, 6))

# Create one scatter point per "dot"
for f, c, n in zip(freqs, counts, dots):
    plt.scatter(np.full(n, f), np.random.normal(c, c*0.02, size=n),
            alpha=0.7, s=30, c=[plt.cm.viridis((f - freqs.min()) / (freqs.max() - freqs.min()))])


plt.xlabel("Frequency (Hz)")
plt.ylabel("ON Event Count")
plt.title("Frequency Response: ON Event Counts per Recording")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

