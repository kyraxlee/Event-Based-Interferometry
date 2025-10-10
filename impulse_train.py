# The load_events function is adapted from the lectures by Yeshwanth Bethi

# A library for picking the event file
# If you do not want to download tkinter or use this, just replace it with the path to the event file
from tkinter.filedialog import askopenfilenames
event_files = askopenfilenames(title="Select Event File")

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# Load events files adapted from Yeshwanth lecture
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

# Function to save the plot with appropriate filename
def save_plot(fig, filename, event_type):
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    # Search for the frequency in the filename
    base = re.search(r'baseline', filename)
    static = re.search(r'static', filename)
    second = re.search(r'\((2)\)', filename)

    # Save the plot
    output_dir = Path("plots/")
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
    events = load_events(event_file)
    rows = 260
    cols = 346

    # Sort events by timestamp
    events = events[events[:, 2].argsort()]
    # Normalise time to start at 0
    events[:, 0] -= events[:, 0].min()

    '''
    Here's the part for changing the event_type for analysis
    '''
    event_type = 0

    # Define a smaller timeframe for more focused analysis
    timeframe = 20000

    # Filter events based on event_type
    focused_events = np.array([e for e in events if e[3] == (event_type if event_type != 0 else e[3])])
    
    # Limit the events to the timeframe
    focused_events = np.array([e for e in focused_events if e[0] <= timeframe])

    # Accumulate events into heatmap
    heatmap = np.zeros((rows, cols))
    for t, x, y, pol in focused_events:
        x, y = int(x), int(y)
        heatmap[y, x] += pol  # y=row, x=column

    # Moving spatial window
    window_size = 10  # e.g., 20x20 pixels
    max_sum = 0
    best_window = (0, 0)

    # Scan through the heatmap to find the window with the highest event counts
    for y in range(rows - window_size + 1):
        for x in range(cols - window_size + 1):
            if event_type == 1:
                roi_sum = np.sum(heatmap[y:y+window_size, x:x+window_size])
            else:
                roi_sum = np.sum(abs(heatmap[y:y+window_size, x:x+window_size]))
            if roi_sum > max_sum:
                max_sum = roi_sum
                best_window = (x, y)

    # Set the window coordinates
    x_bottom = best_window[0]
    x_top = x_bottom + window_size - 1
    y_bottom = best_window[1]
    y_top = y_bottom + window_size - 1

    # Filter the events to only those within the ROI
    roi_events = np.array([e for e in focused_events if x_bottom <= e[1] <= x_top and y_bottom <= e[2] <= y_top])

    # Temporal binning the events
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    else:
        frequency = 120

    # Set a static frequency of 500 Hz for more features
    # Try 0.5 ms or 500 microsecond later
    frequency = 100000  # = 1ms or 1000 microsecond


    print(f"Number of events at {frequency} Hz:", roi_events.shape)

    # Set timestep to 1 microsecond to get each event as a Dirac Impulse
    timestep = 1

    # Create variables for Dirac Impulse
    n_bins = int(np.ceil(roi_events[:, 0].max() / timestep)) + 1
    dirac = np.zeros(n_bins)
    accumulated_events = np.zeros(n_bins)
    signal = 0
    for t, x, y, pol in roi_events:
        dirac[int(t // timestep)] = pol
        signal += pol if pol != 0 else 0
    
    # Plot the Dirac Impulses
    plt.stem(dirac, markerfmt="")
    plt.title("Event data as Dirac Impulses")
    plt.xlabel("Time (us)")
    plt.ylabel("Polarity")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.show()
    #save_plot(plt, event_file, event_type)
    plt.close()