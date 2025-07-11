import numpy as np
import pandas as pd

def assess_symmetry(landmarks):
    """
    Compute symmetry scores between left and right landmarks
    for shoulder, elbow, wrist, hip, knee, and ankle in all frames.

    Args:
        landmarks (pd.DataFrame): Multi-index columns with ('x', idx), ('y', idx)
            indexed by frame.

    Returns:
        pd.DataFrame: DataFrame with columns:
            ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'], indexed by frame.
    """
    # Define landmark indices for each joint (Mediapipe pose indices)
    landmark_indices = {
        'shoulder': {'left': 11, 'right': 12},
        'elbow':    {'left': 13, 'right': 14},
        'wrist':    {'left': 15, 'right': 16},
        'hip':      {'left': 23, 'right': 24},
        'knee':     {'left': 25, 'right': 26},
        'ankle':    {'left': 27, 'right': 28},
    }

    # Initialize output DataFrame
    symmetry_df = pd.DataFrame(index=landmarks.index)

    for joint, sides in landmark_indices.items():
        left_idx = sides['left']
        right_idx = sides['right']

        left_x = landmarks[('x', left_idx)]
        left_y = landmarks[('y', left_idx)]
        right_x = landmarks[('x', right_idx)]
        right_y = landmarks[('y', right_idx)]

        # Compute Euclidean distance between left and right landmark per frame
        symmetry = np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)

        symmetry_df[joint] = symmetry

    return symmetry_df




# def assess_symmetry(landmarks, joint):
#     """
#     Compute symmetry between left and right sides of a specified joint.

#     Parameters:
#         landmarks (DataFrame): DataFrame with MultiIndex columns ('x'/'y', landmark_index).
#         joint (str): One of 'shoulder', 'elbow', 'wrist'.

#     Returns:
#         pd.Series: Symmetry score per frame.
#     """
#     # Define mapping of joints to landmark indices (single point per side)
#     landmark_indices = {
#         'shoulder': {'left': 12, 'right': 11},
#         'elbow':    {'left': 14, 'right': 13},
#         'wrist':    {'left': 16, 'right': 15},
#         'hip':      {'left': 24, 'right': 23},
#         'knee':     {'left': 26, 'right': 25},
#         'ankle':    {'left': 28, 'right': 27},
#     }
    
#     if joint not in landmark_indices:
#         raise ValueError(f"Joint '{joint}' not recognized. Choose from {list(landmark_indices.keys())}.")
    
#     left_idx = landmark_indices[joint]['left']
#     right_idx = landmark_indices[joint]['right']
    
#     # Extract coordinates
#     left_x = landmarks[('x', left_idx)]
#     left_y = landmarks[('y', left_idx)]
#     right_x = landmarks[('x', right_idx)]
#     right_y = landmarks[('y', right_idx)]
    
#     # Compute Euclidean distance between left and right landmarks for each frame
#     symmetry_score = ((left_x - right_x).abs() + (left_y - right_y).abs())
    
#     return symmetry_score

