import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
import numpy as np

# 导入数据
data = {
    "Cd": [0.144, 0.148, 0.158, 0.164, 0.165, 0.194, 0.197, 0.319, 0.335, 0.392, 0.416, 0.469, 0.516, 0.534, 0.612,
          0.603, 0.664, 0.701, 0.75, 0.768, 0.777, 0.787, 0.815, 0.839, 0.841, 0.864, 0.977, 1.035, 1.267, 1.267,
          1.497, 1.654, 1.756, 2.235],
    "Chl": [35.874179, 35.622200, 33.052215, 34.742400, 33.058003, 32.197719, 32.837320, 23.099920, 23.650960,
            24.725040, 22.001520, 24.748800, 21.693600, 28.733300, 23.453720, 27.401880, 22.515200, 21.717880,
            21.204720, 19.909920, 17.822200, 18.800400, 21.240640, 18.190800, 22.336240, 20.206720, 19.804360,
            16.652110, 15.969760, 18.629080, 17.475640, 17.417080, 16.422000, 18.402760]
}

# 创建一个数据框
df = pd.DataFrame(data)

# 计算IQR和异常值的界限
Q1_Cd = df['Cd'].quantile(0.25)
Q3_Cd = df['Cd'].quantile(0.75)
IQR_Cd = Q3_Cd - Q1_Cd

Q1_Chl = df['Chl'].quantile(0.25)
Q3_Chl = df['Chl'].quantile(0.75)
IQR_Chl = Q3_Chl - Q1_Chl

# 根据IQR确定异常值的范围
lower_bound_Cd = Q1_Cd - 1.5 * IQR_Cd
upper_bound_Cd = Q3_Cd + 1.5 * IQR_Cd

lower_bound_Chl = Q1_Chl - 1.5 * IQR_Chl
upper_bound_Chl = Q3_Chl + 1.5 * IQR_Chl

# 过滤掉异常值
filtered_Cd = df['Cd'][(df['Cd'] >= lower_bound_Cd) & (df['Cd'] <= upper_bound_Cd)]
filtered_Chl = df['Chl'][(df['Chl'] >= lower_bound_Chl) & (df['Chl'] <= upper_bound_Chl)]

# 设置字体样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# 设置更学术化的箱线图样式
sns.set_style("whitegrid", {'axes.grid': False})

# 创建一个新的figure对象
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# 设置更学术化的箱线图样式
sns.set_style("whitegrid", {'axes.grid': False})

# Cd的箱线图
ax[0, 0].boxplot(df['Cd'], sym="o", widths=0.3, vert=True)
ax[0, 0].set_title('Cadmium', fontweight='bold', color='black', family='Times New Roman')
ax[0, 0].grid(False)

# 设置边框样式
for _, spine in ax[0, 0].spines.items():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# Chl的箱线图
ax[0, 1].boxplot(df['Chl'], sym="o", widths=0.3, vert=True)
ax[0, 1].set_title('Chlorophyll', fontweight='bold', color='black', family='Times New Roman')
ax[0, 1].grid(False)

# 设置边框样式
for _, spine in ax[0, 1].spines.items():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# Cd的直方图和正态分布拟合曲线
ax[1, 0].hist(filtered_Cd, bins='auto', density=True, color="skyblue", alpha=0.7, edgecolor='black')
sns.kdeplot(filtered_Cd, color="green", ax=ax[1, 0])
xmin, xmax = ax[1, 0].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, filtered_Cd.mean(), filtered_Cd.std())
ax[1, 0].plot(x, p, 'k', linewidth=2)
ax[1, 0].set_title('Cadmium', fontweight='bold', color='black', family='Times New Roman')
ax[1, 0].grid(False)

# 设置边框样式
for _, spine in ax[1, 0].spines.items():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# Chl的直方图和正态分布拟合曲线
ax[1, 1].hist(filtered_Chl, bins='auto', density=True, color="skyblue", alpha=0.7, edgecolor='black')
sns.kdeplot(filtered_Chl, color="green", ax=ax[1, 1])
xmin, xmax = ax[1, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, filtered_Chl.mean(), filtered_Chl.std())
ax[1, 1].plot(x, p, 'k', linewidth=2)
ax[1, 1].set_title('Chlorophyll', fontweight='bold', color='black', family='Times New Roman')
ax[1, 1].grid(False)

# 设置边框样式
for _, spine in ax[1, 1].spines.items():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 调整子图间距
plt.subplots_adjust(wspace=0.3, hspace=0.4)
# 保存图形
plt.savefig("boxplot.png", dpi=600)
# 显示图形
plt.show()

