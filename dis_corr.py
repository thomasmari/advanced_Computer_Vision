import cv2
import time
from pose_ana_corr import get_pushup_state
import mediapipe as mp

# ---------------- SETUP ----------------
print("Push up counter")
feedback_interval = float(input("Feedback Interval (seconds, e.g., 2.5): ").strip())
source = input("test or webcam: ").strip().lower()

# Video source
cap = cv2.VideoCapture(0 if source == "webcam" else "data/push_up5.mp4")
# cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("📸 Starting camera. Press 'q' to quit.")

# State machine variables
prev_state = None
pushup_count = 0
last_feedback_time = 0
rep_state = "idle"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (640,480))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    results = pose_model.process(frame_rgb)

    overlay_text_state = f"State: 'unknown'"
    overlay_text_count = f"Count: 0"

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Convert landmarks
        landmarks = {
            idx: (int(lm.x * w), int(lm.y * h), lm.z, lm.visibility)
            for idx, lm in enumerate(results.pose_landmarks.landmark)
        }

        # Push-up state logic
        current_state = get_pushup_state(landmarks)

        if current_state == "down":
            rep_state = "down"

        elif current_state == "up":
            if rep_state == "down":
                pushup_count += 1
                rep_state = "up"

        prev_state = current_state

        overlay_text_state = f"State: {current_state or 'unknown'}"
        overlay_text_count = f"Count: {pushup_count}"

        # Feedback text
        current_time = time.time()
        if current_time - last_feedback_time >= feedback_interval:
            print(overlay_text_state)
            print(overlay_text_count)
            last_feedback_time = current_time

    # Overlay feedback
    cv2.putText(frame, overlay_text_state, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, overlay_text_count, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Push ups", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"✅ Session Ended. Final count: {pushup_count}")
