# Noise Analysis 
This branch contains additional, focused noise-analysis code developed for the **Event-based Interferometry for Acoustic Imaging and Signal Reconstruction** project. The scripts implement per-pixel and ROI-level analyses to quantify spatial noise, temporal consistency, and signal-to-noise ratio (SNR) from event-camera recordings of laser-reflected surface vibrations.

## Scripts
Each script follows a consistent processing pipeline, comprising the following steps:
1. **Load event data files** from the experimental datasets.  
2. **Save output plots** as `.png` files for record-keeping and figure generation.  
3. **Filter events by polarity**, selecting ON, OFF, or BOTH event types for targeted analysis.  
4. **Identify and extract the Region of Interest (ROI)** corresponding to the primary interferometric activity zone.


The individual script functionality is as follow:
- event_counts.py: Count events to compute all events and compare ROI events vs total to show how much of the signal energy is localized in the ROI
- event_consistency.py: Compute ROI temporal consistency via standard deviations per pixel to determine regular/irregular firing rates
- counts_and_consistency.py: Consolidate results from event_counts.py and event_consistency.py. and flags ROIs as noisy or reliable
- spatial_noise.py: Compute spatial noise heatmaps and per-pixel spatial noise score
- temporal_noise.py: Measure temporal noise using mismatch fractions and related metrics
- SNR.py: Compute per-pixel and ROI SNR using the Moeys et al. formulation adapted to ROI baselines

## Viewing all Plots and Outputs
Only a selection of representative plots are shown in the final report.  To reproduce *all* plots and numerical outputs for every dataset, simply run the scripts in this repository. All figures and CSV files will automatically save into their respective pre-defined directories. This allows full reproducibility of the spatial, temporal, and SNR analyses described in the report. Or, to just view all plots and see the raw data, access the following drive: https://studentuwsedu-my.sharepoint.com/:f:/g/personal/22171055_student_westernsydney_edu_au/EsIdPH_ZvrNBhD0BDbotPNgB75sBmdd5dn-h1H9UA8vC6w?e=euzjva

