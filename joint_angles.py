import pandas as pd
import numpy as np


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def calculate_joint_angles(landmarks):
    
    # Define the landmarks indices for each joint (3 points needed for angle calculation)
    # Each joint angle requires three landmarks: e.g., for elbow angle: shoulder-elbow-wrist
    # So for each joint, specify the three landmarks in order (point1, vertex, point3)
    joints = {
        'shoulder_left':  (24, 12, 14),     # (example indices for shoulder angle: hip-shoulder-elbow)
        'shoulder_right': (23, 11, 13),
        'elbow_left':     (12, 14, 16),     # shoulder, elbow, wrist
        'elbow_right':    (11, 13, 15),
        'wrist_left':     (14, 16, 18),     # (example indices for wrist angle: elbow-wrist-hand)
        'wrist_right':    (13, 15, 17),
        'hip_left':       (12, 24, 26),
        'hip_right':      (11, 23, 25),
        'knee_left':      (24, 26, 28),
        'knee_right':     (23, 25, 27),
        'ankle_left':     (26, 28, 32),
        'ankle_right':    (25, 27, 31),
    }
    
    # Initialize result DataFrame with exactly the columns we want
    joint_angles = pd.DataFrame(index=landmarks.index, columns=list(joints.keys()))
    
    for col, (p1_idx, vertex_idx, p3_idx) in joints.items():
        for frame in landmarks.index:
            try:
                p1 = np.array([
                    landmarks.loc[frame, ('x', p1_idx)],
                    landmarks.loc[frame, ('y', p1_idx)]
                ])
                vertex = np.array([
                    landmarks.loc[frame, ('x', vertex_idx)],
                    landmarks.loc[frame, ('y', vertex_idx)]
                ])
                p3 = np.array([
                    landmarks.loc[frame, ('x', p3_idx)],
                    landmarks.loc[frame, ('y', p3_idx)]
                ])
                angle = calculate_angle(p1, vertex, p3)
            except Exception:
                angle = np.nan
            
            joint_angles.at[frame, col] = angle

    # Reset index to have 'frame' as column
    joint_angles = joint_angles.reset_index()
    
    return joint_angles

