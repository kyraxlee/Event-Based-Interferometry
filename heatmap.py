# A library for picking the event file
# If you do not want to download tkinter or use this, just replace it with the path to the event file
from tkinter.filedialog import askopenfilename
event_file = askopenfilename(title="Select Event File")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load events files adapted from Yeshwanth lecture
def load_events_from_text(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Events file not found: {file_path}")
    
    data = []
    with p.open('r') as f:
        # Strip the first line if it's a header
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('t'):
                continue
            parts = line.split(sep=',')
            if len(parts) != 4:
                continue
            t, x, y, pol = parts
            t = float(t)
            x = int(x)
            y = int(y)
            pol = -1 if int(pol) == 0 else 1
            # Reverse x and y cause the plotting is flipped for some reason
            data.append((y, x, t, pol))
    if not data:
        raise ValueError("No valid events parsed from text file.")
    events_np = np.asarray(data, dtype=np.float64)
    
    rows = 260
    cols = 346
    # sort by time (column 2)
    events_np = events_np[events_np[:, 2].argsort()]
    return events_np, rows, cols

events, rows, cols = load_events_from_text(event_file)

events = events[events[:, 2].argsort()]

print("Number of events:", events.shape[0])

heatmap = np.zeros((rows, cols))
for x, y, t, p in events:
    x, y = int(x), int(y)
    heatmap[x, y] += abs(p)

# This coordinate is reverse
print("Hottest pixel: ", np.unravel_index(np.argmax(heatmap), heatmap.shape))
# Plot the heatmap
plt.imshow(heatmap, cmap="viridis")
plt.colorbar()
plt.show()
plt.close()