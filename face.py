import cv2
import numpy as np
import pandas as pd
import time

prototxt_rgb = r".\model\rgb.prototxt"
caffemodel_rgb = r".\model\rgb.caffemodel"
net_rgb = cv2.dnn.readNetFromCaffe(
    prototxt=prototxt_rgb, caffeModel=caffemodel_rgb)

def detect_faces(frame, min_confidence=0.5):
    (h, w) = frame.shape[:2]    # 獲取影片幀尺寸
    # 預處理幀以進行人臉檢測
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net_rgb.setInput(blob)  # 設定模型的輸入
    detections = net_rgb.forward()  # 執行前向傳播以進行人臉檢測
    faces = []  # 初始化一個列表來存儲檢測到的人臉座標

    # 遍歷所有檢測結果
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # 過濾掉信心值過低的檢測結果
        if confidence < min_confidence:
            continue

        # 計算邊界框的座標
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype("int")

        # 將檢測到的人臉座標添加到列表中
        faces.append((x0, y0, x1 - x0, y1 - y0))
    return faces

def extract_face_color(frame, x, y, w, h):
    face_roi = frame[y:y+h, x:x+w]  # 裁剪出臉部區域
    mean_color = np.mean(face_roi, axis=(0, 1)) # 計算臉部區域的平均顏色
    return mean_color

def main():
    cap = cv2.VideoCapture(r'Z:\thermal_IRB_after_dataset\Sub_1_front_nomask_1\Sub_1_front_nomask_1_30fps_2023-06-30 19_30_00.avi')


    df = pd.DataFrame(columns=["B", "G", "R"])
    # 創建一個空的 DataFrame 以存儲數據
    start_time = time.time()    # 時間初始化

    while True:
        ret, frame = cap.read() # 讀取視頻幀
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 鏡像翻轉影像，第二個參數為1表示水平翻轉

        detected_faces = detect_faces(frame)    # 呼叫人臉檢測函數以獲取人臉座標

        # 繪製檢測到的每一張人臉的邊界框並提取膚色ROI
        for (x, y, w, h) in detected_faces:
            x0 = max(0, x - 10)
            y0 = max(0, y - 10)
            x1 = min(frame.shape[1], x + w + 10)
            y1 = min(frame.shape[0], y + h + 10)

            # 提取膚色ROI並計算平均顏色
            face_color = extract_face_color(frame, x0, y0, x1 - x0, y1 - y0)
            print("Face Color (BGR):", face_color)

            b, g, r = face_color
            df = pd.concat([df, pd.DataFrame({"B": [b], "G": [g], "R": [r]})], ignore_index=True)

            # Create a mask based on skin color
            lower_skin = np.array([0, 20, 70], dtype="uint8")
            upper_skin = np.array([20, 255, 255], dtype="uint8")
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask only for the face region
            face_mask = np.zeros_like(hsv_frame[:, :, 0])
            face_mask[y0:y1, x0:x1] = 255

            # Combine the skin color mask and the face region mask
            skin_mask = cv2.bitwise_and(cv2.inRange(hsv_frame, lower_skin, upper_skin), face_mask)

            # Apply the mask to the frame
            skin_extracted = cv2.bitwise_and(frame, frame, mask=skin_mask)

            # Display the skin-extracted frame
            cv2.imshow("Skin Extracted", skin_extracted)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        current_time = time.time()
        elapsed_time = current_time - start_time    # 添加時間文本
        cv2.putText(frame, f"Time: {int(elapsed_time)}s", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Real-Time Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    df.to_csv("test.csv", index=False)

main()
