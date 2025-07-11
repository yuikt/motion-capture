import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joint_angles import calculate_joint_angles
from symmetry_analysis import assess_symmetry
from smoothness_metrics import compute_jerk
import os, glob, csv
from pathlib import Path

df = pd.read_csv('Motion_centroids_landmarks/daria_run_ct_lm.csv')

# Prepare a dictionary to hold landmarks by frame:
# {frame_number: {landmark_id: (x, y)} }
landmarks_by_frame = {}

# Number of landmarks: assuming 33 landmarks
num_landmarks = 33

for idx, row in df.iterrows():
    # Use idx as frame number
    frame = idx
    landmarks = {}
    for i in range(1, num_landmarks + 1):
        x = row[f'x{i}']
        y = row[f'y{i}']
        landmarks[i] = (x, y)
    
    landmarks_by_frame[frame] = landmarks
print(landmarks_by_frame)

landmark_list = []
frame_list = []
x_list = []
y_list = []
for frame, lm_dict in landmarks_by_frame.items():
    for lm_id, (x, y) in lm_dict.items():
        frame_list.append(frame)
        landmark_list.append(lm_id)
        x_list.append(x)
        y_list.append(y)

landmark_df = pd.DataFrame({
    'frame': frame_list,
    'landmark': landmark_list,
    'x': x_list,
    'y': y_list
})

# Now pivot into wide format with multi-columns (x,y) per landmark
landmark_pivot = landmark_df.pivot(index='frame', columns='landmark', values=['x','y'])

# Calculate joint angles (you must adapt this function to accept landmark_pivot)
joint_angles = calculate_joint_angles(landmark_pivot)

# Assess posture symmetry
symmetry = assess_symmetry(landmark_pivot)

# Compute motion smoothness (jerk)
jerk_score = compute_jerk(landmark_pivot)

print("Average Joint Angles:\n", joint_angles.mean())
print("Symmetry Scores:\n", symmetry)
print("Average Jerk (Smoothness Score):", jerk_score.mean())


# Plot joint angles
# Define joints and colors
joints = ['elbow', 'shoulder', 'wrist']
colors = [('blue', 'red'), ('green', 'orange'), ('purple', 'brown')]
for joint, (c_left, c_right) in zip(joints, colors):
    plt.figure(figsize=(6, 4))
    plt.plot(joint_angles[f'{joint}_left'], label=f'{joint.capitalize()} Left', color=c_left)
    plt.plot(joint_angles[f'{joint}_right'], label=f'{joint.capitalize()} Right', color=c_right)
    plt.title(f"{joint.capitalize()} Angles Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plot symmetry distance
plt.figure(figsize=(12, 6))
for joint in symmetry.columns:
    plt.plot(symmetry.index, symmetry[joint], label=joint.capitalize())

plt.xlabel('Frame')
plt.ylabel('Symmetry Distance (Euclidean)')
plt.title('Symmetry Distances Between Left and Right Joints Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

