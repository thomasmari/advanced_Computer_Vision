# Author: Thomas Mari
# Last updated: 23 Sept.2025

import cv2 #to read and process images
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np


#constantes
nb_landmark = 33 

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setting up the Pose model for videos.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, model_complexity=1)



def frame_to_row(img):
    """
    Extracts pose landmark coordinates from an image frame and returns them as a NumPy array.
    This function processes the input image using a pose estimation model (`pose_img`), extracts the (x, y, z) coordinates
    of each detected landmark, and returns them in a NumPy array of shape (nb_landmark, 3). The x and y coordinates are
    scaled to the image width and height, respectively, while the z coordinate is scaled to the image width.
    Args:
        img (np.ndarray): The input image frame as a NumPy array (H, W, C).
    Returns:
        np.ndarray: A NumPy array of shape (nb_landmark, 3) containing the (x, y, z) coordinates of each landmark.
    Example:
        >>> row = frame_to_row(image)
        >>> print(row.shape)
        (33, 3)  # if nb_landmark is 33
    Note:
        - Requires global variables: `pose_img` (pose estimation model) and `nb_landmark` (number of landmarks).
        - Assumes that `results.pose_landmarks.landmark` is available after processing the image.
    """

    height, width, _ = img.shape
    results = pose_img.process(img)
    frame_landmark_values = np.empty((nb_landmark,3)) # 3 for x,y,z
    # Iterate over the detected landmarks.
    i=0
    if results.pose_landmarks is None:
        return np.zeros((nb_landmark, 3))  # or handle as you need
    for landmark in results.pose_landmarks.landmark:
        # frame_landmark_values[i] = ((int(landmark.x * width), int(landmark.y * height),
                            # (landmark.z * width))) ##version calibree
        frame_landmark_values[i] = (landmark.x,landmark.y,landmark.z)
        i+=1
    return(frame_landmark_values)

def video_to_array(video_path):
    """
    Converts a video file into a NumPy array of extracted landmarks for each frame.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        np.ndarray: A NumPy array where each row corresponds to the landmarks extracted from a video frame.

    Raises:
        FileNotFoundError: If the video file cannot be opened.
        Exception: If frame processing fails.

    Example:
        >>> landmarks = video_to_array("input_video.mp4")
        >>> print(landmarks.shape)
        (num_frames, num_landmarks)

    Note:
        This function requires the `cv2` (OpenCV) and `numpy` libraries, as well as a `frame_to_row` function
        that extracts landmarks from a single RGB frame.
    """    
    cap = cv2.VideoCapture(video_path)
    landmarks_np = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_np.append(frame_to_row(img_rgb))
    return(np.array(landmarks_np))


### test
if __name__=="__main__":
    video_path = "/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/data/chute_ouvrier.mp4"
    res  = video_to_array(video_path)
    print(res.shape)