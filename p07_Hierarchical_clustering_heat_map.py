import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl


# 设置全局字体为 "Times New Roman"
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'
# 加载数据
file_path = "光谱特征参数+Cd和Chl34.xlsx" # 请替换为你的文件路径
df1 = pd.read_excel(file_path)


# 对数据进行归一化
scaler = MinMaxScaler()

df1_normalized = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)


# 创建聚类热图
cluster_grid = sns.clustermap(df1_normalized, cmap="coolwarm", method='average', linewidths=0.5, figsize=(18, 10), cbar_pos=None, tree_kws={"linewidth": 2})


# 获取热图的坐标轴
ax = cluster_grid.ax_heatmap


# 创建颜色条的坐标轴
cbar_ax_positioned_right = cluster_grid.fig.add_axes([ax.get_position().x1 - 0.03, ax.get_position().y0, .02, ax.get_position().height])


# 创建颜色条
plt.colorbar(ax.get_children()[0], cax=cbar_ax_positioned_right)


# 调整子图布局以适应画布
plt.subplots_adjust(right=0.92)  # 调整右边距以适应颜色条


plt.title("Cluster Heatmap", pad=90)
# 保存图像为600 dpi的PNG文件
cluster_grid.savefig("Clustering_heat_map.png", dpi=600)

plt.show()