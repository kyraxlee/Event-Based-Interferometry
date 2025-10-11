import re
from tkinter.filedialog import askopenfilenames
event_files = askopenfilenames(title="Select Event File")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import csv



# First load event data from text file(s)
def load_events_from_text(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Events file not found: {file_path}")
    
    data = []
    with p.open('r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments (#), or header lines starting with 't'
            if not line or line.startswith('#') or line.startswith('t'):
                continue
            
            parts = line.split(',')
            if len(parts) != 4:
                continue  # Skip malformed lines
            
            t, x, y, pol = parts
            t = float(t)   # timestamp
            x = int(x)     # x coordinate
            y = int(y)     # y coordinate
            
            # Polarity: convert 0 → -1 (OFF), 1 → +1 (ON)
            pol = -1 if int(pol) == 0 else 1
            
            # Reverse x and y for correct orientation in display or analysis
            data.append((y, x, t, pol))
    
    if not data:
        raise ValueError("No valid events parsed from text file.")
    
    # Convert list to numpy array and sort by timestamp
    events_np = np.asarray(data, dtype=np.float64)
    events_np = events_np[events_np[:, 2].argsort()]
    
    # Define sensor dimensions (DAVIS346 = 260×346)
    rows, cols = 260, 346
    return events_np, rows, cols



# Process each selected event file
# Prepare CSV output for event summary
output_csv = Path("plots/event_count_summary.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)
write_header = not output_csv.exists()  # Write header only once if file doesn’t exist

for event_file in event_files:

    # Load events for current file
    events, rows, cols = load_events_from_text(event_file)

    # Extract acoustic frequency from filename (e.g., “80Hz”)
    match = re.search(r'(\d+)Hz', event_file)
    frequency = int(match.group(1)) if match else 80  # Default to 80 Hz if not found

    print(f"\n=== Processing file: {Path(event_file).name} ===")
    print(f"Frequency: {frequency} Hz")
    print(f"Total number of events: {events.shape[0]}")

    event_type = 0  # 1 = ON events only, -1 = OFF events only, 0 = both types

    
    # Build spatial heatmap of event activity
    heatmap = np.zeros((rows, cols))
    for x, y, t, p in events:
        x, y = int(x), int(y)
        if p == event_type or event_type == 0:
            heatmap[x, y] += p  # Accumulate polarity-weighted event count

    # Identify pixel with the highest activity (magnitude of event count)
    hottest_x, hottest_y = np.unravel_index(np.argmax(np.abs(heatmap)), heatmap.shape)
    print(f"Hottest pixel location: (x={hottest_x}, y={hottest_y})")

   
    # Metric 1: Count events to compute all events and compare ROI events vs total
    # Purpose: shows how much of the signal energy is localized in the ROI
    roi_half_size = 5  # defines ROI radius around the hottest pixel
    roi_x_start = max(hottest_x - roi_half_size, 0)
    roi_x_end   = min(hottest_x + roi_half_size, rows)
    roi_y_start = max(hottest_y - roi_half_size, 0)
    roi_y_end   = min(hottest_y + roi_half_size, cols)

    # Extract ROI region
    roi_mask = np.zeros((rows, cols), dtype=bool)
    roi_mask[roi_x_start:roi_x_end, roi_y_start:roi_y_end] = True

    # Compute counts
    total_events = events.shape[0]
    on_events_total  = np.sum(events[:, 3] == 1)
    off_events_total = np.sum(events[:, 3] == -1)

    # Identify which events fall inside ROI
    in_roi = np.array([
        roi_mask[int(x), int(y)] for x, y, _, _ in events
    ])
    roi_events_all = np.sum(in_roi)
    roi_events_on  = np.sum(in_roi & (events[:, 3] == 1))
    roi_events_off = np.sum(in_roi & (events[:, 3] == -1))

    # Ratios relative to total
    roi_ratio_all = roi_events_all / total_events
    roi_ratio_on  = roi_events_on / on_events_total if on_events_total > 0 else np.nan
    roi_ratio_off = roi_events_off / off_events_total if off_events_total > 0 else np.nan

    # Print metrics
    print(f"\n--- Metric 1: Event Count ---")
    print(f"Total Events        : {total_events}")
    print(f"ON Events           : {on_events_total}")
    print(f"OFF Events          : {off_events_total}")
    print(f"ROI Events (All)    : {roi_events_all}")
    print(f"ROI Events (ON)     : {roi_events_on}")
    print(f"ROI Events (OFF)    : {roi_events_off}")
    print(f"ROI Ratio (All)     : {roi_ratio_all:.4f}")
    print(f"ROI Ratio (ON)      : {roi_ratio_on:.4f}")
    print(f"ROI Ratio (OFF)     : {roi_ratio_off:.4f}")


    # Save to CSV 
    with output_csv.open('a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Filename", "Frequency_Hz",
                "Total_Events", "ON_Events", "OFF_Events",
                "ROI_Events_All", "ROI_Events_ON", "ROI_Events_OFF",
                "ROI_Ratio_All", "ROI_Ratio_ON", "ROI_Ratio_OFF"
            ])
            write_header = False

        writer.writerow([
            Path(event_file).name, frequency,
            total_events, on_events_total, off_events_total,
            roi_events_all, roi_events_on, roi_events_off,
            roi_ratio_all, roi_ratio_on, roi_ratio_off
        ])
