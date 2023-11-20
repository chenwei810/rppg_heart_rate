import pandas as pd
import matplotlib.pyplot as plt

# 从CSV文件读取
# df = pd.read_csv("test.csv")
df = pd.read_csv("face_color_signals.csv")

# 提取颜色通道数据
B = df["B"]
G = df["G"]
R = df["R"]

# 创建时间序列（假设每一帧的时间间隔为1秒）
time_series = range(len(B))

# 创建一个新的图形，分成三个子图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))

# 绘制B、G、R颜色通道的折线图在不同的子图中
ax1.plot(time_series, B, label="Blue", color="blue")
ax2.plot(time_series, G, label="Green", color="green")
ax3.plot(time_series, R, label="Red", color="red")

# 添加标签和标题

ax1.set_ylabel("Blue Color Value")
ax2.set_ylabel("Green Color Value")
ax3.set_xlabel("Time (frames)")
ax3.set_ylabel("Red Color Value")

# 添加图例
ax1.legend()
ax2.legend()
ax3.legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
