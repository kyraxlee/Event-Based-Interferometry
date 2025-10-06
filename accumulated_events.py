# A library for picking the event file
# If you do not want to download tkinter or use this, just replace it with the path to the event file
from tkinter.filedialog import askopenfilenames
event_files = askopenfilenames(title="Select Event File")

import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

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

# Function to save the plot with appropriate filename
def save_plot(fig, filename, event_type):
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
    output_dir = Path("plots/total_events")
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
    events, rows, cols = load_events_from_text(event_file)

    events = events[events[:, 2].argsort()]

    # Set the bounds for the square we want to focus in.
    # The commented value is (almost) the entire laser region
    #x_upper, x_lower = 150, 100              
    #y_upper, y_lower = 260, 210              
    x_lower, x_upper = 215, 224
    y_lower, y_upper = 89, 98

    # Filter the events according to the bounds
    focused_events = np.array([e for e in events if x_lower <= e[0] < x_upper and y_lower <= e[1] < y_upper])
    focused_events[:, 2] -= focused_events[0, 2] # Normalize time to start at 0
    focused_events[:, 0] -= x_lower             # Normalize x to start at 0
    focused_events[:, 1] -= y_lower             # Normalize y to start at 0

    # Set the rows and cols for plotting
    rows = x_upper - x_lower + 1
    cols = y_upper - y_lower + 1

    # Intialize variables for plotting 
    # Timestamps in event data is in microseconds

    # Search for the frequency in the filename
    match = re.search(r'(\d+)Hz', event_file)

    if match:
        frequency = int(match.group(1))
    else:
        frequency = 80

    print(f"Number of events at {frequency} Hz:", focused_events.shape[0])
    
    # Timestep is set to twice the frequency for Nyquist
    timestep = int(1000000 / (frequency * 2)) # microseconds

    # For plotting
    surface = np.zeros((rows, cols))
    total_events = np.zeros(int(focused_events[-1, 2] // timestep) + 1)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    figure = plt.imshow(surface, cmap='viridis')

    '''
    Here's the part for changing the event_type for plotting
    '''
    event_type = 1
    index = 0
    bin = 0

    for x, y, t ,p in focused_events :
        x, y = int(x), int(y)
        bin_index = int(t // timestep)

        if event_type == p or event_type == 0:
            surface[x, y] += p

        if bin_index > bin: 
            figure.set_data(surface)
            fig.canvas.draw_idle()
            # Comment this line to skip the animation
            #plt.pause(0.1)
            # Get the total number of events
            total_events[bin] = np.sum((surface))
            # Reset the surface to display the next frame
            # Can implement a decay factor here instead
            surface = np.zeros((rows, cols)) 
            # Go to next timestep
            bin = bin_index
    # Remove the last bin as it is incomplete
    total_events = total_events[:-1]
    plt.close()
    
    # Make a time vector for x axis
    time_vector = np.arange(0, total_events.shape[0] * timestep, timestep)
    # Plot the total events over time
    plt.figure(figsize=(8, 4))
    plt.plot(time_vector, total_events, linestyle = '-', color='b')
    plt.xlabel("Timestep")
    plt.ylabel("Number of events")
    plt.title(f"Accumulated events per timestep at {frequency} Hz")
    plt.show()
    #save_plot(plt, event_file, event_type)
    plt.close()