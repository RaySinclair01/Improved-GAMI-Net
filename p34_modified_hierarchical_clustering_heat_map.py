
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

# Setting the font globally (this might not work in this environment, but should work locally)
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'

# Load the data
df1 = pd.read_csv("光谱特征参数+Cd和Chl无组名34.csv")

# Normalize the data
scaler = MinMaxScaler()
df1_normalized = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns, index=df1.index)

# Create the clustered heatmap rotated 90 degrees counter-clockwise with color bar on the right
plt.figure(figsize=(7, 8))
sns.set(font="Times New Roman", font_scale=1)
cluster_grid = sns.clustermap(df1_normalized.transpose(), figsize=(7, 8),cmap="coolwarm", row_cluster=True, dendrogram_ratio=(0.28,0.08),col_cluster=True, cbar_pos=(0.84, 0.06, 0.025, 0.6), xticklabels=False, linewidths=0.5, linecolor="lightgray", tree_kws={"linewidth": 1.6})
plt.setp(cluster_grid.ax_row_dendrogram.lines, linewidth=1.6)  # Make the lines in the clustering tree thicker

# Show the feature names
cluster_grid.ax_heatmap.set_yticklabels(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("Clustering_heat_map2.png", dpi=600)
plt.show()
