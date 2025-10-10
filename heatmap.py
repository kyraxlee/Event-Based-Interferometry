# The load_events_from_text function is adapted from the lectures by Yeshwanth Bethi

# A library for picking the event file
# If you do not want to download tkinter or use this, just replace it with the path to the event file
import re
from tkinter.filedialog import askopenfilenames
event_files = askopenfilenames(title="Select Event File")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
matplotlib.use('Agg')

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

def save_plot(fig, filename, event_type):
    # Search for the frequency in the filename
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    else:
        frequency = 80
    base = re.search(r'baseline', filename)
    static = re.search(r'static', filename)
    second = re.search(r'\((2)\)', filename)

        # Save the plot
    output_dir = Path("plots/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory based on event type
    if event_type == 1:
        output_dir = output_dir / "ON"
    elif event_type == -1:
        output_dir = output_dir / "OFF"
    else:
        output_dir = output_dir / "BOTH"

    output_dir.mkdir(exist_ok=True) 

    # Save the plot with appropriate name
    if base:
        plt.savefig(output_dir / f"baseline.png")
    elif static:
        plt.savefig(output_dir / f"static.png")
    else:
        if second:
            plt.savefig(output_dir / f"{frequency}Hz(2).png")
        else:
            plt.savefig(output_dir / f"{frequency}Hz.png")
    return

# Loop through each selected event file
for event_file in event_files:
    events, rows, cols = load_events_from_text(event_file)

    # Sort events by timestamp
    events = events[events[:, 2].argsort()]

    # Normalise time to start at 0
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    else:
        frequency = 80

    print(f"Number of events at {frequency} Hz:", events.shape[0])

    event_type = 0 # 1 for ON events, -1 for OFF events, 0 for both

    # Accumulate events into heatmap
    heatmap = np.zeros((rows, cols))
    for x, y, t, p in events:
        x, y = int(x), int(y)
        if p == event_type or event_type == 0:
            heatmap[x, y] += p

    # This coordinate is reverse
    hottest_x, hottest_y = np.unravel_index(np.argmax(np.abs(heatmap)), heatmap.shape)
    print("Hottest pixel:", (hottest_x, hottest_y))

    # Draw a small box around the hottest pixel
    box_size = 5
    plt.figure(figsize=(10, 8))
    plt.title(f"Event Heatmap at {frequency} Hz for {'ON' if event_type == 1 else 'OFF' if event_type == -1 else 'BOTH'} Events")

    # Plot the heatmap
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar()
    # Draw a rectangle around the hottest pixel
    box_size = 5
    rect = Rectangle(
        (hottest_y - box_size, hottest_x - box_size),  # x,y are swapped for display coordinates
        2 * box_size,  # width
        2 * box_size,  # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    plt.gca().add_patch(rect)
    plt.show()
    #save_plot(plt, event_file, event_type)
    plt.close()