import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
import re

# Choose the file for analysis
from tkinter.filedialog import askopenfilenames
event_files = askopenfilenames(title="Select Event File")

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
            pol = int(pol)  # 1 = ON, 0 = OFF
            data.append((t, x, y, pol))
    events_np = np.array(data, dtype=np.float64)
    events_np = events_np[events_np[:, 0].argsort()]  # Sort by time
    return events_np

# Function to save the plot with appropriate filename
def save_plot(fig, filename, event_type, zoom=False):
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    else:
        frequency = 80
    # Search for the frequency in the filename
    base = re.search(r'baseline', filename)
    static = re.search(r'static', filename)
    second = re.search(r'\((2)\)', filename)

        # Save the plot
    output_dir = Path("plots/spatial window")
    if zoom:
        output_dir = output_dir / "zoomed"
        
    output_dir.mkdir(parents=True, exist_ok=True)

    if event_type == 1:
        output_dir = output_dir / "ON"
    elif event_type == -1:
        output_dir = output_dir / "OFF"
    else:
        output_dir = output_dir / "BOTH"

    output_dir.mkdir(exist_ok=True) 

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

for event_file in event_files:
    # Load events
    events = load_events(event_file)

    # Filter for region of interest and ON events
    x_lower, x_upper = 0, 346
    y_lower, y_upper = 0, 260

    focused_events = np.array([
        e for e in events 
        if x_lower <= e[1] < x_upper and y_lower <= e[2] < y_upper and e[3] == 1
    ])
    focused_events[:, 1] -= x_lower  # Normalize x
    focused_events[:, 2] -= y_lower  # Normalize y

    rows = y_upper - y_lower
    cols = x_upper - x_lower

    print("Number of ON events:", focused_events.shape[0])

    event_type = 1

    # Accumulate events into heatmap
    heatmap = np.zeros((rows, cols))
    for t, x, y, pol in focused_events:
        x, y = int(x), int(y)
        if event_type == pol or event_type == 0:
            heatmap[y, x] += pol  # y=row, x=column

    # Moving spatial window
    window_size = 20  # e.g., 20x20 pixels
    max_sum = 0
    best_window = (0, 0)

    for y in range(rows - window_size + 1):
        for x in range(cols - window_size + 1):
            roi_sum = np.sum(heatmap[y:y+window_size, x:x+window_size])
            if roi_sum > max_sum:
                max_sum = roi_sum
                best_window = (x, y)

    x_bottom = best_window[0]
    x_top = x_bottom + window_size - 1
    y_bottom = best_window[1]
    y_top = y_bottom + window_size - 1

    print(f"Window with highest activity:")
    print(f"x_bottom = {x_bottom}, x_top = {x_top}")
    print(f"y_bottom = {y_bottom}, y_top = {y_top}")

    # Plot 1: accumulated heatmap with the best window highlighted
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar(label='Number of ON events')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Accumulated ON events heatmap")

    # Draw rectangle around best window
    rect = plt.Rectangle((x_bottom, y_bottom), window_size, window_size,
                        edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)
    #plt.show()
    save_plot(plt, event_file, event_type)
    plt.close()



    # Plot 2: Zoomed-in around ROI 
    zoom_margin = 2  # pixels around the window for context
    x1 = max(0, x_bottom - zoom_margin)
    x2 = min(cols, x_top + zoom_margin)
    y1 = max(0, y_bottom - zoom_margin)
    y2 = min(rows, y_top + zoom_margin)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar(label='Number of ON events')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Zoomed-In View Around ROI")
    plt.xlim(x1, x2)
    plt.ylim(y2, y1)  # flip for correct orientation
    rect_zoom = plt.Rectangle((x_bottom, y_bottom), window_size, window_size,
                            edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect_zoom)
    plt.show()
    save_plot(plt, event_file, event_type, zoom=True)
    plt.close()