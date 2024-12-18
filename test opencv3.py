import cv2
import mediapipe as mp
import numpy as np
import socket



# Setup socket client
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('10.238.60.6', 9000))
server_socket.listen(5)
client_socket, address = server_socket.accept()
print(f"Connected to {address}")

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open a connection to the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_height, frame_width = frame.shape[:2]

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Calculate midpoints
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        upper_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        upper_mid_y = (left_shoulder.y + right_shoulder.y) / 2

        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        lower_mid_x = (left_hip.x + right_hip.x) / 2
        lower_mid_y = (left_hip.y + right_hip.y) / 2

        body_center_x = (upper_mid_x + lower_mid_x) / 2
        body_center_y = (upper_mid_y + lower_mid_y) / 2

        # Clip and convert to pixel coordinates
        body_center_x = np.clip(body_center_x, 0, 1)
        body_center_y = np.clip(body_center_y, 0, 1)
        body_center_screen_x = int(body_center_x * frame_width)
        body_center_screen_y = int(body_center_y * frame_height)

        # Draw center point
        cv2.circle(frame, (body_center_screen_x, body_center_screen_y), 10, (0, 255, 0), -1)

        # Calculate relative Y position
        relative_y = 2 * (((body_center_screen_y - 720) / 720) * -5 - 2.5)
        formatted_y = round(relative_y, 7)

        message = str(formatted_y).encode('utf-8')
        client_socket.sendall(message)

        # Display Y coordinate
        cv2.putText(frame, f"Body Center Y: {formatted_y}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Camera Capture with Neck Position', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
server_socket.close()
