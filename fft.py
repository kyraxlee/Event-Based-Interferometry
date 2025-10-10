# A library for picking the event file
# If you do not want to download tkinter or use this, just replace it with the path to the event file
# The load_events function is adapted from the lectures by Yeshwanth Bethi

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
    output_dir = Path("plots/500Hz event counts/")
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

# Loop through each event file
for event_file in event_files:
    events = load_events(event_file)
    rows = 260
    cols = 346

    # Sort events by timestamp
    events = events[events[:, 0].argsort()]

    '''
    Here's the part for changing the event_type for analysis
    '''
    event_type = 1

    # Filter events based on event_type
    focused_events = np.array([e for e in events if e[3] == (event_type if event_type != 0 else e[3])])
    # Normalise the events to start at 0
    focused_events[:, 0] -= focused_events[:, 0].min()

    # Accumulate events into heatmap
    heatmap = np.zeros((rows, cols))
    for t, x, y, pol in focused_events:
        x, y = int(x), int(y)
        heatmap[y, x] += pol  # y=row, x=column

    # Moving spatial window
    window_size = 10  # e.g., 20x20 pixels
    max_sum = 0
    best_window = (0, 0)

    # Scan the heatmap to find the window with the highest event counts
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

    # Filter events to the ROI
    roi_events = np.array([e for e in focused_events if x_bottom <= e[1] <= x_top and y_bottom <= e[2] <= y_top])
    # Pick a smaller time frame for a zoom in view
    roi_events = np.array([e for e in roi_events if e[0] <= 200000])

    # Temporal binning the events
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    else:
        frequency = 120

    # Set a static frequency of 500 Hz for better FFT analysis
    frequency = 500

    print(f"Number of events at {frequency} Hz:", roi_events.shape)

    # Sample the events at twice the supposed frequency  for Nyquist
    timestep = int(1000000 / (frequency * 2)) # microseconds

    # Temporal binning and count events in each bin
    n_bins = int(np.ceil(roi_events[:, 0].max() / timestep)) + 1
    accumulated_events = np.zeros(n_bins)
    for t, x, y, pol in roi_events:
        bin_idx = int(t // timestep)
        accumulated_events[bin_idx] += pol
    
    # Create a time vector for plotting
    time_vector = np.arange(n_bins) * timestep

    # Plot the event counts in each time bin
    plt.plot(time_vector, accumulated_events, label=f"{frequency} Hz")
    plt.xlabel("Time (us)")
    plt.ylabel("Number of events in ROI")
    plt.title(f"Events Count for {frequency} Hz in ROI")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #save_plot(plt, event_file, event_type)
    plt.close()

    # Offset the accumulated events to be zero-mean
    accumulated_events -= np.mean(accumulated_events)
    fft_values = np.fft.fft(accumulated_events)
    fft_freq = np.fft.fftfreq(len(accumulated_events), d=timestep/1000000)
    fft_magnitude = np.abs(fft_values)
    plt.plot(fft_freq[:len(fft_freq)//2], fft_magnitude[:len(fft_magnitude)//2])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"FFT of Accumulated Events for {(timestep)} us in ROI")
    plt.show()
    #save_plot(plt, event_file, event_type)
    plt.close()