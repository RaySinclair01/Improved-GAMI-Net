import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
# 加载新的数据文件
file_path_updated = '光谱特征参数+Cd和Chl加组名34.xlsx'
data_updated = pd.read_excel(file_path_updated)
# 定义用于绘制的数据
groups = ['clean', 'low', 'medium', 'high']
columns = data_updated.columns[:21]
# 定义存储相关系数的字典
correlations_chl_updated = {'clean': [], 'low': [], 'medium': [], 'high': []}
correlations_cd_updated = {'clean': [], 'low': [], 'medium': [], 'high': []}

# 按照group列的clean、low、medium、high分组计算相关系数
for group in ['clean', 'low', 'medium', 'high']:
    subset = data_updated[data_updated['group'] == group]
    for col in subset.columns[:21]:
        corr_chl = np.corrcoef(subset[col], subset['Chl'])[0, 1]
        corr_cd = np.corrcoef(subset[col], subset['Cd'])[0, 1]
        correlations_chl_updated[group].append(corr_chl)
        correlations_cd_updated[group].append(corr_cd)
# 将相关系数转换为 DataFrame，以便进行热图可视化
correlations_chl_df = pd.DataFrame(correlations_chl_updated, index=columns)
correlations_cd_df = pd.DataFrame(correlations_cd_updated, index=columns)
# 定义颜色映射，使颜色从clean到high依次变深
colors = {'clean': 'lightblue', 'low': 'skyblue', 'medium': 'deepskyblue', 'high': 'dodgerblue'}

# 创建一个2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(20, 18), gridspec_kw={'width_ratios': [2, 1]})

# 分组条形图表示与Chl的相关系数
for group in groups:
    correlations = [correlations_chl_updated[group][i] for i in range(21)]
    axes[0, 0].bar(columns, correlations, color=colors[group], alpha=0.5, label=group)
axes[0, 0].set_title('Correlations with Chl (Bar Plot)')
axes[0, 0].set_xlabel('Columns')
axes[0, 0].set_ylabel('Correlation')
axes[0, 0].tick_params(axis='x', rotation=90)
axes[0, 0].legend()

# 分组条形图表示与Cd的相关系数
for group in groups:
    correlations = [correlations_cd_updated[group][i] for i in range(21)]
    axes[1, 0].bar(columns, correlations, color=colors[group], alpha=0.5, label=group)
axes[1, 0].set_title('Correlations with Cd (Bar Plot)')
axes[1, 0].set_xlabel('Columns')
axes[1, 0].set_ylabel('Correlation')
axes[1, 0].tick_params(axis='x', rotation=90)
axes[1, 0].legend()

# 热图表示与Chl的相关系数
sns.heatmap(correlations_chl_df, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Correlations with Chl (Heatmap)')
axes[0, 1].set_xlabel('Group')
axes[0, 1].set_ylabel('Columns')

# 热图表示与Cd的相关系数
sns.heatmap(correlations_cd_df, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlations with Cd (Heatmap)')
axes[1, 1].set_xlabel('Group')
axes[1, 1].set_ylabel('Columns')

plt.tight_layout()
# 保存图形，并设置分辨率为600 dpi
plt.savefig("group_bar_heat_correl.png", dpi=600)  # 替换成你要保存的图片路径和文件名
plt.show()

