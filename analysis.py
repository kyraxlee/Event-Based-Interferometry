# A library for picking the event file
# If you do not want to download tkinter or use this, just replace it with the path to the event file
from tkinter.filedialog import askopenfilename
event_file = askopenfilename(title="Select Event File")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
num_events = events.shape[0]

print("Number of events:", num_events)
timestamps = np.zeros((rows, cols), dtype=np.float64) - np.inf
polarity = np.zeros((rows, cols), dtype=np.float64)

time_tau = 10000.0

def get_exponential_timesurf(polarity, timestamps, t, tau=5.0):
    return np.exp(- (t - timestamps)/tau) * polarity

exponential_timesurface = get_exponential_timesurf(polarity, timestamps, 0, time_tau)

fig = plt.figure(figsize=(10, 8))
figure = plt.imshow(exponential_timesurface, cmap='viridis', vmin=-1, vmax=1)

timestamps = np.zeros((rows, cols), dtype=np.float64) - np.inf
polarity = np.zeros((rows, cols), dtype=np.float64)
last_displayed_t = -1.0

for event_idx in range(num_events):
    x, y, t, p = events[event_idx]
    x, y = int(x), int(y)
    timestamps[x, y] = t
    polarity[x, y] = p

    if event_idx % 100000 == 0:
        last_displayed_t = t

        exponential_timesurface = get_exponential_timesurf(polarity, timestamps, t, time_tau)
        
        figure.set_data(exponential_timesurface)
        fig.canvas.draw_idle()
        plt.pause(0.1)