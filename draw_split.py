import pandas as pd
import matplotlib.pyplot as plt

# 從CSV文件讀取
df = pd.read_csv("face_color_signals.csv")

# 提取顏色通道數據
R = df["R"]
G = df["G"]
B = df["B"]

# 創建時間序列（假設每一幀的時間間隔為1秒）
time_series = range(len(B))

# 創建一個新的圖形，分成三個子圖
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))

# 繪製R、G、B顏色通道的折線圖在不同的子圖中
ax1.plot(time_series, R, label="Red", color="red")
ax2.plot(time_series, G, label="Green", color="green")
ax3.plot(time_series, B, label="Blue", color="blue")

# 添加標籤和標題
ax1.set_ylabel("Red Color Value")
ax2.set_ylabel("Green Color Value")
ax3.set_ylabel("Blue Color Value")
ax3.set_xlabel("Time (frames)")

# 添加圖例
ax1.legend()
ax2.legend()
ax3.legend()

# 調整子圖之間的間距
plt.tight_layout()

# 顯示圖形
plt.show()
