import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 读取第一个sheet的数据
file_path = "Hyperspectral_datatemp.xlsx"
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

# 创建画布和子图
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
plt.subplots_adjust(wspace=0.3)

# 指定颜色的渐变
color_order = ["clean", "low", "medium", "high"]
palette = sns.color_palette("Blues", n_colors=len(color_order))

# 为“Cd”列绘制小提琴图
sns.violinplot(x="group", y="Cd", data=data, order=color_order, palette=palette, ax=axes[0])
axes[0].set_title('Cd Distribution', fontsize=18)
axes[0].set_ylabel('Cd Value', fontsize=16)
axes[0].set_xlabel('Group', fontsize=16)

# 为“Chl”列绘制小提琴图
sns.violinplot(x="group", y="Chl", data=data, order=color_order, palette=palette, ax=axes[1])
axes[1].set_title('Chl Distribution', fontsize=18)
axes[1].set_ylabel('Chl Value', fontsize=16)
axes[1].set_xlabel('Group', fontsize=16)

# 添加图例并去掉外框
legend_labels = [plt.Line2D([0], [0], color=color, lw=4) for color in palette]
legend0 = axes[0].legend(legend_labels, color_order, title='Groups', fontsize=14)
legend1 = axes[1].legend(legend_labels, color_order, title='Groups', fontsize=14)
legend0.get_frame().set_edgecolor('none')
legend1.get_frame().set_edgecolor('none')
# 设置坐标轴为纯黑色，并增加线条粗细
for ax in axes:
    for _, spine in ax.spines.items():
        spine.set_color('black')
        spine.set_linewidth(2)
plt.savefig("Cd+Chl_grouped_data_distribution.png", dpi=600)
plt.show()