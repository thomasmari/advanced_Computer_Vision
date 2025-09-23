import numpy as np

# Mediapipe landmark index mapping
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16
}

def get_point(landmarks, name):
    idx = LANDMARKS[name]
    if idx not in landmarks:
        return None
    return (landmarks[idx][0], landmarks[idx][1])

def angle(a, b, c):
    if None in (a, b, c):
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return None
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def get_pushup_state(landmarks):
    """
    Returns the current state: 'up', 'down', or None if data is invalid.
    """
    left_elbow_angle = angle(get_point(landmarks, "left_shoulder"),
                             get_point(landmarks, "left_elbow"),
                             get_point(landmarks, "left_wrist"))

    right_elbow_angle = angle(get_point(landmarks, "right_shoulder"),
                              get_point(landmarks, "right_elbow"),
                              get_point(landmarks, "right_wrist"))

    if left_elbow_angle is None or right_elbow_angle is None:
        return None

    avg_angle = (left_elbow_angle + right_elbow_angle) / 2

    if avg_angle < 90:
        return "down"
    elif avg_angle > 160:
        return "up"
    else:
        return "in_between"
