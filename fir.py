import pandas as pd
import numpy as np
from scipy.signal import firwin, lfilter, find_peaks
import matplotlib.pyplot as plt

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