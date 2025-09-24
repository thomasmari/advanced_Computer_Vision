#author : Thomas Mari
#Debug : Jerome Delaunay et Thomas Mari
#Last updated : 24 september 2025

import cv2
import mediapipe as mp
from preprocessing import estimPose_img
from extract_landmark import video_to_array, frame_to_row
from fall_features_extraction import cha_table_features
from features_extraction import FeaturesExtraction

# Initializing mediapipe pose class.features 
mp_pose = mp.solutions.pose

# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setting up the Pose model for videos.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, model_complexity=1)
# Initializing mediapipe drawing class to draw landmarks on specified image.
mp_drawing = mp.solutions.drawing_utils

def ui_features_from_path(video_path:str):
    #extract all features
    res  = video_to_array(video_path)
    print(res.shape)
    features = cha_table_features(res)

    cap = cv2.VideoCapture(video_path)
    i=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_img.process(img_rgb)

        # Draw the pose landmarks if detected
        if results.pose_landmarks:
            # # draw all landmark
            # mp_drawing.draw_landmarks(
            #     frame, 
            #     results.pose_landmarks, 
            #     mp_pose.POSE_CONNECTIONS
            # )
            #Draw specific landmarks
            h, w, _ = frame.shape
            indices = [0, 11, 12, 23, 24, 15, 16, 27, 28]  # nez, epaules, hanches, poignets, chevilles
            for idx in indices:
                lm = results.pose_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Red circles
            #Draw Features
            for features_idx in [1,2]:            
                x,y = features[i,features_idx,0:2]
                x, y = int(x*w), int(y * h)
                #postion
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # green circle
                #speed vector 
                vy = int(features[i,features_idx,4]*h/10)
                cv2.arrowedLine(frame, (x, y), (x, y+vy), (255, 0, 0), 3, tipLength=0.2)

        # Resize frame for display
        frame_resized = cv2.resize(frame, (960, 540))
        cv2.imshow("MediaPipe Video", frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
        i+=1

    cap.release()
    cv2.destroyAllWindows()

def ui_features_from_webcam():
    #init webcam feed
    cap = cv2.VideoCapture(0)
    #init feature extraction instance
    fe = FeaturesExtraction()
    i=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_img.process(img_rgb)
        fe.compute_feature_row(frame_to_row(img_rgb))
        frame_feature = fe.get_previous_feature_row()
        # Draw the pose landmarks if detected
        if results.pose_landmarks:
            h, w, _ = frame.shape
            indices = [0, 11, 12, 23, 24, 15, 16, 27, 28]  # nez, epaules, hanches, poignets, chevilles
            for idx in indices:
                lm = results.pose_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Red circles
            #Draw Features
            for features_idx in [1,2]:            
                x,y = frame_feature[features_idx,0:2]
                x, y = int(x*w), int(y * h)
                #postion
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # green circle
                #speed vector 
                vy = int(frame_feature[features_idx,4]*h/10)
                cv2.arrowedLine(frame, (x, y), (x, y+vy), (255, 0, 0), 3, tipLength=0.2)

        # Resize frame for display
        frame_resized = cv2.resize(frame, (960, 540))
        cv2.imshow("MediaPipe Video", frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
        i+=1

    cap.release()
    cv2.destroyAllWindows()

# {    # features indices
#     nez = 0
#     centre_epaules = 1
#     centre_hanches = 2
#     poignet_gauche = 3
#     poignet_droit = 4
#     cheville_gauche = 5
#     cheville_droite = 6
#     }


# {    # landmarks indices
#     nez = 0
#     epaules = [11, 12] # gauche_droite
#     hanches = [23, 24] # gauches_droites
#     poignet_gauche = 15
#     poignet_droit = 16
#     cheville_gauche = 27
#     cheville_droite = 28
#     }

if __name__ == "__main__":
    video_path = "data/video_chute/istockphoto-1066783428-640_adpp_is.mp4"    
    # ui_features_from_path(video_path)
    ui_features_from_webcam()