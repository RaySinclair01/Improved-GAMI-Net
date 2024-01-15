from scipy.stats import iqr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 读取第一个sheet的数据
file_path = "屈原光谱数据3-fa1.xlsx"
data = pd.read_excel(file_path, sheet_name=0)

# 设置图表样式和字体
sns.set_style("white")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 指定颜色的渐变
color_order = ["clean", "low", "medium", "high"]
palette = sns.color_palette("Blues", n_colors=len(color_order))

# 计算每个组的统计量
group_stats = data.groupby('group').agg(
    Cd_median=('Cd', 'median'),
    Chl_median=('Chl', 'median'),
    Cd_iqr=('Cd', iqr),
    Chl_iqr=('Chl', iqr),
    Cd_std=('Cd', 'std'),
    Chl_std=('Chl', 'std'),
).reset_index()

# 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
plt.subplots_adjust(wspace=0.3)

# 分别为Cd和Chl设置标注间距
label_spacing = {'Cd': 0.08, 'Chl': 1.2}

# 绘制小提琴图和统计量
for idx, col in enumerate(["Cd", "Chl"]):
    sns.violinplot(x="group", y=col, data=data, order=color_order, palette=palette, ax=axes[idx])

    # 标注统计量
    for i, group in enumerate(color_order):
        stats = group_stats.loc[group_stats['group'] == group]
        median = stats[f'{col}_median'].values[0]
        iqr_value = stats[f'{col}_iqr'].values[0]
        std = stats[f'{col}_std'].values[0]

        axes[idx].text(i, median - label_spacing[col], f'Median: {median:.2f}', ha='center', va='center', fontsize=10,
                       color='red')
        axes[idx].text(i, median - label_spacing[col] * 2, f'IQR: {iqr_value:.2f}', ha='center', va='center',
                       fontsize=10, color='green')
        axes[idx].text(i, median - label_spacing[col] * 3, f'STD: {std:.2f}', ha='center', va='center', fontsize=10,
                       color='blue')

    # 设置标题和标签
    axes[idx].set_title(f'{col} Distribution', fontsize=18)
    axes[idx].set_ylabel(f'{col} Value', fontsize=16)
    axes[idx].set_xlabel('Group', fontsize=16)

    # 添加图例并去掉外框
    legend_labels = [plt.Line2D([0], [0], color=color, lw=4) for color in palette]
    legend = axes[idx].legend(legend_labels, color_order, title='Groups', fontsize=14)
    legend.get_frame().set_edgecolor('none')

    # 设置坐标轴为纯黑色，并增加线条粗细
    for _, spine in axes[idx].spines.items():
        spine.set_color('black')
        spine.set_linewidth(2)

# 保存并展示图表
plt.savefig("Cd+Chl_grouped_data_distribution_with_adjusted_spacing.png", dpi=600)
plt.show()

group_stats