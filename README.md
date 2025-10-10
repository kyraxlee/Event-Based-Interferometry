# Event-Based Interferometry for Acoustic Imaging and Signal Reconstruction
## Scripts
This repository has the Python scripts used to analyse the data coollected for the Event-based Interferometry for Acoustic Imaging and Signal Reconstruction project.
All of the scripts in the repository follows these common intial processing steps:
1. Load the event data files.
2. A function to save the generated plots as a png file.
3. Filter the data by the event polarities with either ON, OFF, or BOTH types of event.


The individual script functionality is as follow:
- accumulated_events.py: Seperate the events into uniform time bin and count the number of events.
- impulse_train.py: Graph every event in the form of an impulse train to see the sparsity of events.
- event_frequencies.py: Show the total event counts per region ROI across all frequencies.
- fft.py: Count the number of events per time bin in the ROI and perform FFT on them.
- heatmap.py: Create a heatmap of the whole 346x260 frame.
- spatial_window.py: Scan over the entire frame heatmap to find the region with the most event counts.
- temporal_window_boxplot.py: Generate the boxplot for the mean and standard deviation of all ON temporal event counts per global temporal bin across
frequencies in the selected ROI.
- temporal_window_lineplot.py: Generate the lineplot for the total event counts per gloabal temporal bin across frequencies in the selected ROI.


## Data
The data as well as the generated graphs are in the following link: https://studentuwsedu-my.sharepoint.com/:f:/g/personal/20599616_student_westernsydney_edu_au/EoQLunsXG4VNp3fumEM0NRABhCZolOas3k1rSoWsDaszWA?e=6VW0Nf
