import cv2
import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter, find_peaks
import matplotlib.pyplot as plt

# 讀取模型配置和權重
prototxt_rgb = r".\model\rgb.prototxt"
caffemodel_rgb = r".\model\rgb.caffemodel"
net_rgb = cv2.dnn.readNetFromCaffe(
    prototxt=prototxt_rgb, caffeModel=caffemodel_rgb)

# 定義函數以偵測人臉
def detect_faces(frame, min_confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net_rgb.setInput(blob)  # 進行前向傳播以檢測人臉
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

# 定義函數以提取人臉區域的顏色
def extract_face_color(frame, x, y, w, h):
    face_roi = frame[y:y+h, x:x+w]
    mean_color = np.mean(face_roi, axis=(0, 1))
    return mean_color

# 讀取視訊檔案
cap = cv2.VideoCapture(r'Z:\thermal_IRB_after_dataset\Sub_1_front_nomask_1\Sub_1_front_nomask_1_30fps_2023-06-30 19_30_00.avi')
# 創建 DataFrame 以存儲顏色和相關數據
df = pd.DataFrame(columns=["R", "G", "B", "S1", "S2", "Std_S1", "Std_S2", "PR_raw", "PR_normalized"])

# 初始化 frame 計數器、S1 和 S2 的 buffer 以及 buffer 大小
frame_count = 0  # 初始化 frame 計數器
s1_buffer = []  # 保存 S1 的 buffer
s2_buffer = []  # 保存 S2 的 buffer
buffer_size = 90  # buffer 大小
pr_raw_values = []  # 保存 PR_raw 的 buffer

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 偵測人臉
    detected_faces = detect_faces(frame)

    for (x, y, w, h) in detected_faces:
        x0 = max(0, x - 10)
        y0 = max(0, y - 10)
        x1 = min(frame.shape[1], x + w + 10)
        y1 = min(frame.shape[0], y + h + 10)

        face_color = extract_face_color(frame, x0, y0, x1 - x0, y1 - y0)
        # print("Face Color (RGB):", face_color)

        r, g, b = face_color
        # 將顏色數據添加到 DataFrame
        df = pd.concat([df, pd.DataFrame({"R": [r], "G": [g], "B": [b], "S1": [r], "S2": [g], "PR_raw": [0], "PR_normalized": [0]})], ignore_index=True)

        # 對 RGB 通道應用矩陣運算，POS演算
        matrix_operation_result = np.array([
            [0, 1, -1],
            [-2, 1, 1]
        ]).dot([r, g, b])

        # print("Matrix Operation Result:", matrix_operation_result)

        # 將 S1 和 S2 加入 buffer
        s1_buffer.append(matrix_operation_result[0])
        s2_buffer.append(matrix_operation_result[1])

        # 檢查緩衝區是否超過指定的大小
        if len(s1_buffer) > buffer_size:
            # 移除最早加入的frame
            s1_buffer.pop(0)
            s2_buffer.pop(0)

        # 將結果添加到 DataFrame
        df.loc[df.index[-1], 'S1'] = matrix_operation_result[0]
        df.loc[df.index[-1], 'S2'] = matrix_operation_result[1]

        # 每90個 frame 計算一次 std_s1 和 std_s2
        if frame_count > 90:
            std_s1 = np.std(s1_buffer)
            std_s2 = np.std(s2_buffer)
            # print(f"Standard Deviation of S1 (Frame {frame_count}): {std_s1}")
            # print(f"Standard Deviation of S2 (Frame {frame_count}): {std_s2}")

            # 計算 S1 標準差除以 S2 標準差的結果
            alpha = std_s1 / std_s2

            if len(pr_raw_values) > buffer_size:
                pr_raw_values.pop(0)
            # 計算 PR_raw
            PR_raw = std_s1 + (alpha * std_s2)
            pr_raw_values.append(PR_raw)

            # 計算 PR_raw 平均值
            PR_mean = np.mean(pr_raw_values)
            # 計算 PR_raw 標準差
            PR_std = np.std(pr_raw_values)
            # 計算 PR_normalized
            if PR_std != 0:
                PR_normalized = (PR_raw - PR_mean) / PR_std
            else:
                # 如果 PR_std 為零，請根據您的需求設置一個預設值，這裡設置為零
                PR_normalized = 0

            # Append the results to the DataFrame
            df.loc[df.index[-1], 'Std_S1'] = std_s1
            df.loc[df.index[-1], 'Std_S2'] = std_s2
            df.loc[df.index[-1], 'PR_raw'] = PR_raw
            df.loc[df.index[-1], 'PR_mean'] = PR_mean
            df.loc[df.index[-1], 'PR_std'] = PR_std
            df.loc[df.index[-1], 'PR_normalized'] = PR_normalized

        # 在視訊畫面上畫框框
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        frame_count += 1  # 每次 frame 更新計數器
        print(frame_count)

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

    # 侵蝕操作
    kernel_erode = np.ones((17, 17), np.uint8)
    face_mask = cv2.erode(face_mask, kernel_erode, iterations=2)

    # 膨脹操作
    kernel_dilate = np.ones((1, 1), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel_dilate, iterations=2)

    skin_mask = cv2.bitwise_and(cv2.inRange(hsv_frame, lower_skin, upper_skin), face_mask)
    skin_extracted = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # 顯示實時視窗
    cv2.imshow("Real-Time Face Detection", frame)
    cv2.imshow("Skin Extracted", skin_extracted)

    # 按下 "Q" 鍵時中止
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 輸出處理後的 DataFrame 至 CSV 檔案
df.to_csv("test_with_fir.csv", index=False)

cap.release()
cv2.destroyAllWindows()

# 讀取 CSV 文件
df = pd.read_csv("face_color_signals_normalized.csv")

pr_raw = df["PR_raw"]

# 取出 PR_normalized 資料
pr_normalized = df["PR_normalized"]

# 設計 80 階 FIR 濾波器的係數
order = 80
nyquist = 0.5 * 30  # Nyquist 頻率，這裡假設取樣頻率為 30 Hz
cutoff = [1 / nyquist, 1.67 / nyquist]  # 截至頻率，轉換為正規化頻率

# 計算 FIR 濾波器係數
coefficients = firwin(order, cutoff, pass_zero=False)

# 應用 FIR 濾波器到 PR_normalized 資料
pr_filtered = lfilter(coefficients, 1.0, pr_normalized)

# 將處理完的訊號用 PR_filtered 表示
df["PR_filtered"] = pr_filtered

# 將更新後的 DataFrame 寫回 CSV 檔案
df.to_csv("face_color_signals_normalized.csv", index=False)

# 找到 PR_filtered 曲線上的峰值
peaks, _ = find_peaks(pr_filtered, height=0)
# 計算峰值的總和
peaks_sum = len(peaks)

# 繪製 PR_normalized 和 PR_filtered 曲線
plt.plot(peaks, pr_filtered[peaks], "x", label="Peaks", color="red")
# plt.plot(pr_raw, label="PR_raw")
# plt.plot(pr_normalized, label="PR_normalized")
plt.plot(pr_filtered, label="PR_filtered")
plt.title("PR_normalized and PR_filtered with FIR Filter")
plt.xlabel("Frame")
plt.ylabel("Value")
plt.legend()
print(peaks_sum)
plt.show()