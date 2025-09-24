from extract_landmark import video_to_array
from fall_features_extraction import cha_table_features
import cv2
import numpy as np


### ---------------------------------------------
def compute_torso_angle(shoulders, hips):
    """
    Returns angle (in degrees) between torso vector (shoulders to hips) and vertical axis (y-axis).
    0° = upright, 90° = lying down
    """
    torso_vector = hips - shoulders  # shape: (nb_frames, 3)
    vertical_vector = np.array([0, -1, 0])  # y-axis up

    dot_product = np.sum(torso_vector * vertical_vector, axis=1)
    norms = np.linalg.norm(torso_vector, axis=1) * np.linalg.norm(vertical_vector)
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)

    angles = np.arccos(cos_theta) * 180 / np.pi  # convert to degrees
    return angles  # shape: (nb_frames,)


def detect_fall_improved(features, vertical_axis=1,
                velocity_threshold=2.0, acceleration_threshold=10.0,
                angle_threshold=100.0, height_drop_threshold=80):
    """
    Improved fall detection based on multiple criteria.
    """

    nb_frames = features.shape[0]

    # Extract Y (vertical) position
    shoulder_y = features[:, 1, 0 + vertical_axis]
    hip_y = features[:, 2, 0 + vertical_axis]

    # Extract velocities and accelerations
    shoulder_vy = features[:, 1, 3 + vertical_axis]
    hip_vy = features[:, 2, 3 + vertical_axis]
    shoulder_ay = features[:, 1, 7 + vertical_axis]
    hip_ay = features[:, 2, 7 + vertical_axis]

    print(shoulder_vy, hip_vy, shoulder_ay, hip_ay)

    # Reconstruct positions to compute orientation
    shoulders = features[:, 1, 0:3]
    hips = features[:, 2, 0:3]
    angles = compute_torso_angle(shoulders, hips)  # 0° = upright, 90° = flat
    # print(angles)

    for i in range(20, nb_frames):
        # Condition 1: Sudden downward motion
        fast_downward = (shoulder_vy[i] > velocity_threshold and
                         hip_vy[i] > velocity_threshold and
                         shoulder_ay[i] > acceleration_threshold and
                         hip_ay[i] > acceleration_threshold)
        

        # Condition 2: Torso becomes more horizontal
        horizontal_posture = angles[i] < angle_threshold

        # Condition 3: Drop in torso height
        initial_torso_y = (shoulder_y[i - 19] + hip_y[i - 19]) / 2
        current_torso_y = (shoulder_y[i] + hip_y[i]) / 2
        height_drop = current_torso_y - initial_torso_y  # y increases downward in MediaPipe
        # print(height_drop)

        significant_drop = height_drop > height_drop_threshold

        # if fast_downward and horizontal_posture and significant_drop:
        # if horizontal_posture:
        # if significant_drop:
        if significant_drop and horizontal_posture:
        # if fast_downward:
            return True, i  # Fall detected

    return False, -1


if __name__ == "__main__":
    # video_path = "/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/data/chute_ouvrier.mp4"
    # video_path = "/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/data/chute_walking-trip.mp4"
    video_path = "/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/data/chute_banana-peel.mp4"
    # res  = video_to_array(video_path)
    # print(res.shape)
    # features = cha_table_features(res)
    # print(features.shape)
    # print(features[:, 0, :])


    landmarks = video_to_array(video_path)  # Shape: (nb_frames, 33, 3) from MediaPipe
    features = cha_table_features(landmarks)
    fall_detected, fall_frame = detect_fall_improved(features) #detect_fall(features)

    if fall_detected:
        print(f"Fall detected at frame {fall_frame}")
    else:
        print("No fall detected.")


    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    fall_frame_to_show = fall_frame  # from your detection function

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index == fall_frame_to_show:
            cv2.imshow("Fall Detected Frame", frame)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()
            break

        frame_index += 1

    cap.release()
