import pandas as pd
import matplotlib.pyplot as plt

# # 讀取 CSV 檔案
# df = pd.read_csv("face_color_signals_test_rrrrrr.csv")
# # 繪製 PR_raw 的折線圖
# plt.plot(df["PR_raw"], label="PR_raw")
# plt.title("PR_raw")
# plt.xlabel("Frame")
# plt.ylabel("PR_raw")
# plt.legend()
# plt.show()

# # 讀取 CSV 檔案
# df = pd.read_csv("face_color_signals_test_nor.csv")
# # 繪製 PR_normalized 的折線圖
# plt.plot(df["PR_normalized"], label="PR_normalized")
# plt.title("PR_normalized")
# plt.xlabel("Frame")
# plt.ylabel("PR_normalized")
# plt.legend()
# plt.show()

# 讀取 CSV 檔案
df = pd.read_csv("face_color_signals_filtered.csv")
# 繪製 PR_filtered 的折線圖
plt.plot(df["PR_filtered"], label="PR_filtered")
plt.title("PR_filtered")
plt.xlabel("Frame")
plt.ylabel("PR_filtered")
plt.legend()
plt.show()
