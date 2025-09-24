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

# Set the bounds for the square we want to focus in.
# The commented value is (almost) the entire laser region
x_upper = 150       # 50
x_lower = 100       # 220
y_upper = 260       # 110
y_lower = 210       # 310

# Filter the events according to the bounds
focused_events = np.array([e for e in events if x_lower <= e[0] < x_upper and y_lower <= e[1] < y_upper])
focused_events[:, 2] -= focused_events[0, 2] # Normalize time to start at 0
focused_events[:, 0] -= x_lower             # Normalize x to start at 0
focused_events[:, 1] -= y_lower             # Normalize y to start at 0

# Set the rows and cols for plotting
rows = y_upper - y_lower + 1
cols = x_upper - x_lower + 1

print("Number of events:", focused_events.shape[0])

# Intialize variables for plotting 
# Timestamps in event data is in microseconds
frequency = int(input("Enter the frequency: ")) # Hz
# Timestep is set to twice the frequency for Nyquist
timestep = int(1000000 / (frequency * 2)) # microseconds

# For plotting
surface = np.zeros((rows, cols))
total_events = np.zeros(int(focused_events[-1, 2] // timestep) + 1)

# Plot
fig = plt.figure(figsize=(10, 8))
figure = plt.imshow(surface, cmap='viridis', vmin=-1, vmax=1)
index = 0

for x, y, t ,p in focused_events :
    x, y = int(x), int(y)
    surface[x, y] = p
    if int(t // timestep) == index: 
        figure.set_data(surface)
        fig.canvas.draw_idle()
        # Comment this line to skip the animation
        #plt.pause(0.1)
        # Get the total number of events
        total_events[int(t // timestep)] = np.sum(abs(surface))
        # Reset the surface to display the next frame
        # Can implement a decay factor here instead
        surface = np.zeros((rows, cols)) 
        # Go to next timestep
        index += 1
    
plt.close()

# Plot the total events over time
plt.figure(figsize=(8, 4))
plt.plot(total_events, linestyle = '-', color='b', label="Data")
plt.xlabel("Timestep")
plt.ylabel("Number of events")
plt.title("Number of events per timestep")
plt.show()