import cv2
import mediapipe as mp
from marie_pushups.preprocessing import estimPose_img
import numpy as np

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Setting up the Pose model for videos.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class to draw landmarks on specified image.
mp_drawing = mp.solutions.drawing_utils

#data video
cap = cv2.VideoCapture("data/video_pompe/4945123-uhd_4096_2160_24fps.mp4")
#webcam
# cap = cv2.VideoCapture(0)



def landmark_angle(img_process,i1,i2,i3):
# Get landmark positions (normalized coordinates)
        lm11 = img_process.pose_landmarks.landmark[i1]
        lm13 = img_process.pose_landmarks.landmark[i2]
        lm15 = img_process.pose_landmarks.landmark[i3]

        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape
        x11, y11 = int(lm11.x * w), int(lm11.y * h)
        x13, y13 = int(lm13.x * w), int(lm13.y * h)
        x15, y15 = int(lm15.x * w), int(lm15.y * h)

        # Compute the angle at the elbow (13)
        a = np.array([x11, y11])
        b = np.array([x13, y13])
        c = np.array([x15, y15])

        ba = a - b
        bc = c - b

        # Compute angle in degrees
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return(np.degrees(angle))

def pose_high(img):
    results = pose_img.process(img_rgb)

    # Draw the pose landmarks if detected
    if results.pose_landmarks:
        #left arm 
        angle_deg_left_arm = landmark_angle(results,11,13,15)
        if angle_deg_left_arm >= 160 and angle_deg_left_arm <= 210:
            return True 
        angle_deg_right_arm = landmark_angle(results,12,14,16)
        if angle_deg_right_arm >= 160 and angle_deg_right_arm <= 210:
            return True
    return False 

def pose_low(img):
    results = pose_img.process(img_rgb)

    # Draw the pose landmarks if detected
    if results.pose_landmarks:
        #left arm 
        angle_deg_left_arm = landmark_angle(results,11,13,15)
        if angle_deg_left_arm >= 60 and angle_deg_left_arm <= 100:
            return True 
        angle_deg_right_arm = landmark_angle(results,12,14,16)
        if angle_deg_right_arm >= 60 and angle_deg_right_arm <= 100:
            return True
    return False 


counter = {"last_pose":"OTHER", "count":0}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_img.process(img_rgb)

    # Draw the pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
    # Detect pose type
    pose_type = ""
    if pose_high(img_rgb):
        pose_type = "HIGH"
    elif pose_low(img_rgb):
        pose_type = "LOW"
    else:
        pose_type = "OTHER"
    #updating counter
    if counter['last_pose'] in ['OTHER'] and pose_type in ['HIGH','LOW']:
        counter['last_pose']=pose_type
    if counter['last_pose']=='HIGH' and pose_type=='LOW':
        counter['last_pose']='LOW'
    if counter['last_pose']=='LOW' and pose_type=='HIGH':
        counter['last_pose']='HIGH'
        counter['count'] += 1

    # Display pose type on the frame
    cv2.putText(frame, f"Pose: {pose_type}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 3)
    cv2.putText(frame, f"Counter: {counter}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 3)




    # Resize frame for display
    frame_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("MediaPipe Video", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()