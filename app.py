import streamlit as st
import numpy as np
from PIL import Image

import cv2
import mediapipe as mp
from src.preprocessing import estimPose_img
from src.extract_landmark import video_to_array, frame_to_row
from src.fall_features_extraction import cha_table_features
from src.analytics_classifier import detect_fall_improved_video
from src.features_extraction import FeaturesExtraction #webcam requirement

# Initializing mediapipe pose class.features 
mp_pose = mp.solutions.pose

# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setting up the Pose model for videos.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, model_complexity=1)
# Initializing mediapipe drawing class to draw landmarks on specified image.
mp_drawing = mp.solutions.drawing_utils


def run_webcam_stream():
    

    col1, col2, col3 = st.columns(3)

    with col2:
        st.image("/home/marie.edet@Digital-Grenoble.local/Documents/mod18_acv/part2_projet/FallIsComing.webp", width=200)
        st.session_state.run_webcam = False

        if st.button("Start Webcam", width=200):
            st.session_state.run_webcam = True
    
    FRAME_WINDOW = st.image([])


    cap = cv2.VideoCapture(0)
    fe = FeaturesExtraction()  # Initialize feature extractor once here

    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not available")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_img.process(img_rgb)
        fe.compute_feature_from_frame(img_rgb)
        fall_state, height_drop, significant_drop_state, angles, horizontal_posture_state, fast_downward_state = fe.frame_to_state()

        if results.pose_landmarks:
            h, w, _ = frame.shape
            indices = [0, 11, 12, 23, 24, 15, 16, 27, 28]
            for idx in indices:
                lm = results.pose_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            for features_idx in [1, 2]:
                x, y = fe.get_position()[features_idx, 0:2]
                x, y = int(x * w), int(y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                vy = int(fe.get_velocity()[features_idx, 1] * h / 10)
                cv2.arrowedLine(frame, (x, y), (x, y + vy), (255, 0, 0), 3, tipLength=0.2)

        
        if fall_state:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)
            cv2.putText(frame, "Et c'est la chute !", (60, 320), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3, cv2.LINE_AA)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()

if __name__ == "__main__":
    run_webcam_stream()
