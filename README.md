# Motion_capture

## Pose detection and landmark extraction
Run `motion_capture_landmarks.py` to detect 33 landmarks and calculate their centroids of each video
  * centroid_x, centroid_y, andlist of 33 landmarks of each video frame are stored in folder `Motion_centroids_landmarks`

< NOTE > \
The original video files (folder `WISDM`) used in this project are from the paper:

**Actions as Space-Time Shapes**  
Lena Gorelick, Moshe Blank, Eli Shechtman, Michal Irani, and Ronen Basri

If you use these videos or this dataset, please consider citing the original publication:

> Gorelick, L., Blank, M., Shechtman, E., Irani, M., & Basri, R. (2007). Actions as space-time shapes. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 29(12), 2247–2253. [https://doi.org/10.1109/TPAMI.2007.70711](https://doi.org/10.1109/TPAMI.2007.70711)


## Analysis 
Run `motion_capture_analysis.ipy` for analyzing gait pattern or joint angle estimation

This project analyzes human motion data (e.g., joint angles, symmetry, and smoothness) extracted using Mediapipe. 


## Features

- Joint angle tracking (elbows, knees, shoulders)
- Symmetry analysis of left-right body parts
- Smoothness and jerk score evaluation
- Basic mesh and trajectory visualizations
