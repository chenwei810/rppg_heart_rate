import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

prototxt_rgb = r".\model\rgb.prototxt"
caffemodel_rgb = r".\model\rgb.caffemodel"
net_rgb = cv2.dnn.readNetFromCaffe(
    prototxt=prototxt_rgb, caffeModel=caffemodel_rgb)

def detect_faces(frame, min_confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net_rgb.setInput(blob)
    detections = net_rgb.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < min_confidence:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype("int")
        faces.append((x0, y0, x1 - x0, y1 - y0))
    return faces

def extract_face_color(frame, x, y, w, h):
    face_roi = frame[y:y+h, x:x+w]
    mean_color = np.mean(face_roi, axis=(0, 1))
    return mean_color

cap = cv2.VideoCapture(r'Z:\thermal_IRB_after_dataset\Sub_1_front_nomask_1\Sub_1_front_nomask_1_30fps_2023-06-30 19_30_00.avi')
df = pd.DataFrame(columns=["B", "G", "R"])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))

def update(frame):
    global df
    ret, frame = cap.read()
    if not ret:
        ani.event_source.stop()

    frame = cv2.flip(frame, 1)

    detected_faces = detect_faces(frame)

    for (x, y, w, h) in detected_faces:
        x0 = max(0, x - 10)
        y0 = max(0, y - 10)
        x1 = min(frame.shape[1], x + w + 10)
        y1 = min(frame.shape[0], y + h + 10)

        face_color = extract_face_color(frame, x0, y0, x1 - x0, y1 - y0)
        print("Face Color (BGR):", face_color)

        b, g, r = face_color
        df = pd.concat([df, pd.DataFrame({"B": [b], "G": [g], "R": [r]})], ignore_index=True)

        # 在視訊畫面上畫框框
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # 清空畫布
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # 提取颜色通道数据
    B = df["B"]
    G = df["G"]
    R = df["R"]

    # 创建时间序列（假设每一帧的时间间隔为1秒）
    time_series = range(len(B))

    # 繪製B、G、R颜色通道的直方圖
    ax1.plot(time_series, B, label="Blue", color="blue")
    ax1.set_title("Blue Channel")

    ax2.plot(time_series, G, label="Green", color="green")
    ax2.set_title("Green Channel")

    ax3.plot(time_series, R, label="Red", color="red")
    ax3.set_title("Red Channel")

    # 添加標籤和標題
    ax1.set_ylabel("Blue Color Value")
    ax2.set_ylabel("Green Color Value")
    ax3.set_xlabel("Time (frames)")
    ax3.set_ylabel("Red Color Value")

    # 添加圖例
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # 調整子圖之間的間距
    plt.tight_layout()

    # 提取膚色區域
    lower_skin = np.array([0, 20, 70], dtype="uint8")
    upper_skin = np.array([20, 255, 255], dtype="uint8")
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    face_mask = np.zeros_like(hsv_frame[:, :, 0])
    for (x, y, w, h) in detected_faces:
        x0 = max(0, x - 10)
        y0 = max(0, y - 10)
        x1 = min(frame.shape[1], x + w + 10)
        y1 = min(frame.shape[0], y + h + 10)
        face_mask[y0:y1, x0:x1] = 255

    skin_mask = cv2.bitwise_and(cv2.inRange(hsv_frame, lower_skin, upper_skin), face_mask)
    skin_extracted = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # 顯示實時視窗
    cv2.imshow("Real-Time Face Detection", frame)
    cv2.imshow("Skin Extracted", skin_extracted)

ani = FuncAnimation(fig, update, frames=range(500), interval=100)
plt.show()

cap.release()
cv2.destroyAllWindows()
df.to_csv("face_color_signals.csv", index=False)
