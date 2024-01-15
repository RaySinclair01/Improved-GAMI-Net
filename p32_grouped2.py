import matplotlib.pyplot as plt
import numpy as np
# First, let's read the data from the uploaded Excel file to understand its structure.
import pandas as pd

# Load the first worksheet from the Excel file
file_path = '屈原光谱数据3-fa1.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Filter data based on groups
groups = ['clean', 'low', 'medium', 'high']
# 设置画布和子图，所有子图共享x和y轴的范围
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
axes = axes.ravel()
# Set common axis labels and font properties
font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 26}
# 使用学术的彩色风格
colors = plt.cm.tab10(np.linspace(0, 1, len(df['site'].unique())))

# 遍历每个组并绘制光谱曲线
for i, (ax, group) in enumerate(zip(axes, groups)):
    if i != 0:
        ax.set_ylabel('')  # Remove the y-axis label for the last three subplots
    ax = axes[i]
    group_data = df[df['group'] == group]
    wavelengths = np.arange(325, 1076)  # 325到1075的波长数据
    for idx, (color, (_, row)) in enumerate(zip(colors, group_data.iterrows())):
        ax.plot(wavelengths, row.loc[325:1075], color=color)

    # 设置标题、坐标轴标签和字体
    ax.set_title(f'{group.capitalize()} Group', font_properties)
    ax.set_xlabel('Wavelength (nm)', font_properties)
    if i == 0: ax.set_ylabel('Reflectance', font_properties)
    ax.tick_params(labelsize=11, labelrotation=0)
    # Change the font of tick labels to "Times New Roman" and make it bold
    for label in ax.get_xticklabels():
        label.set_family('serif')
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_family('serif')
        label.set_fontweight('bold')
    # 在坐标内加一个黑框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)



# 调整布局并显示图形
plt.tight_layout()
plt.savefig("grouped.png", dpi=600)  # 替换成你要保存的图片路径和文件名
plt.show()
