import cv2
import numpy as np
import mediapipe as mp
import csv
import os

# Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Hàm lấy Pose và trung điểm giữa hai vai
def getPose(image):
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Chế độ nhẹ, nhanh hơn
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as pose:
        
        image_height, image_width, _ = image.shape
        results = pose.process(image)

        center_x, center_y, center_z = None, None, None
        if results.pose_landmarks:
            # Lấy điểm trung tâm giữa LEFT_SHOULDER và RIGHT_SHOULDER
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            center_x = (left_shoulder.x + right_shoulder.x) / 2
            center_y = (left_shoulder.y + right_shoulder.y) / 2
            center_z = (left_shoulder.z + right_shoulder.z) / 2

            # Vẽ các điểm trên cơ thể và trung điểm
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            center_px = int(center_x * image_width)
            center_py = int(center_y * image_height)
            cv2.circle(image, (center_px, center_py), 5, (0, 0, 255), -1)  # Vẽ điểm trung tâm

        return image, center_x, center_y, center_z

# Cấu hình YOLO
net = cv2.dnn.readNet("YOLO test/yolov3-tiny (1).weights", "YOLO test/yolov3-tiny.cfg")
with open("YOLO test/YOLO-Realtime-Human-Detection/lib/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Bật GPU cho YOLO nếu khả dụng
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Khởi tạo Camera và CSV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

with open(os.path.join(os.getcwd(), "cat.csv"), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['t', 'person', 'center_x', 'center_y', 'center_z'])

# Biến điều khiển
t = 0
frame_count = 0
skip_frames = 2  # Số lượng khung hình muốn bỏ qua
store = []
patient = []
nurse = []

# Xử lý video từ camera
while True:
    _, img = cap.read()
    if not _: break

    # Bỏ qua khung hình để tăng tốc
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    height, width, _ = img.shape

    # YOLO Detection
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if x < 0 or y < 0:
                continue

            person = img[y:y+h, x:x+w]  # Cắt ảnh vùng người
            if len(store) == 0:
                store.append(1)
                patient.append(boxes[i])
            elif len(store) == 1:
                store.append(1)
                nurse.append(boxes[i])
            else:
                label = 'Person 1' if abs(x - patient[-1][0]) < abs(x - nurse[-1][0]) else 'Person 2'
                image, cx, cy, cz = getPose(person)

                # Vẽ khung và thông tin
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

                # Ghi dữ liệu vào CSV
                with open("cat.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([t, label, cx, cy, cz])

                cv2.imshow(label, image)

    t += 1
    cv2.imshow("Camera Input", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
