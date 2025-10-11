# Noise Analysis
This branch is for further noise analysis that has been done in addition to the group project. 
## Scripts
This repository has the Python scripts used specifically for noise analysis to analyse the data coollected for the Event-based Interferometry for Acoustic Imaging and Signal Reconstruction project.
All of the scripts in the repository follows these common initial processing steps:
1. Load the event data files.
2. A function to save the generated plots as a png file.
3. Filter the data by the event polarities with either ON, OFF, or BOTH types of event.


The individual script functionality is as follow:
- event_counts.py: Count events to compute all events and compare ROI events vs total to show how much of the signal energy is localized in the ROI
