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

# Focus on the region the laser is pointing at
focused_events = np.array([e for e in events if 60 <= e[0] < 210 and 120 <= e[1] < 300])
rows = 150
cols = 180
focused_events[:, 2] -= focused_events[0, 2] # Normalize time to start at 0
focused_events[:, 0] -= 60 # Normalize x to start at 0
focused_events[:, 1] -= 120 # Normalize y to start at 0

print("Number of events:", focused_events.shape[0])

# Intialize variables for plotting 
# Timestamps in event data is in microseconds
frequency = 80 # Hz
timestep = np.floor(1000000 / frequency * 2) # microseconds
surface = np.zeros((rows, cols))
total_events = np.zeros(int(focused_events[-1, 2] // timestep) + 1)

# Plot
fig = plt.figure(figsize=(10, 8))
figure = plt.imshow(surface, cmap='viridis', vmin=-1, vmax=1)

for x, y, t ,p in focused_events :
    x, y = int(x), int(y)
    surface[x, y] = p

    if t % timestep == 0:
        total_events = np.sum(np.abs(surface))
        figure.set_data(surface)
        fig.canvas.draw_idle()
        plt.pause(0.1)
        # Reset the surface to display the next frame
        # Can implement a decay factor here instead
        surface = np.zeros((rows, cols))

plt.close()

# Plot the total events over time
print(total_events)