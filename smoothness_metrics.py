import numpy as np

def compute_jerk(landmarks):
    """Calculate jerk (third derivative of position) as smoothness measure."""
    jerk_scores = []

    for lm in landmarks.columns.levels[1]:
        x = landmarks['x'][lm].values
        dx = np.gradient(x)
        ddx = np.gradient(dx)
        dddx = np.gradient(ddx)
        jerk_scores.append(np.mean(np.abs(dddx)))

    return np.array(jerk_scores)
