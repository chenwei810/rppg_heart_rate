import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
# df = pd.read_csv("face_color_signals_filtered_with_FIR.csv")
df = pd.read_csv("face_color_signals_normalized.csv")

# 繪製 PR_filtered 的折線圖
# plt.plot(df["PR_normalized"], label="PR_normalized")
plt.plot(df["PR_filtered"], label="PR_filtered")
plt.title("PR_filtered")
plt.xlabel("Frame")
plt.ylabel("PR_filtered")
plt.legend()
plt.show()
