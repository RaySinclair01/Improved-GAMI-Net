
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties
# 定义多项式拟合函数
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

# 拟合函数并计算R^2值
def fit_function(func, x_data, y_data):
    params, _ = curve_fit(func, x_data, y_data)
    y_pred = func(x_data, *params)
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return params, r_squared

# 根据Cd值分级设置颜色
def assign_color(cd_value):
    if cd_value < 0.2:
        return '#004775'
    elif 0.2 < cd_value < 0.3:
        return '#41A0BC'
    elif 0.3 < cd_value < 0.6:
        return '#F7A947'
    else:
        return '#E45C5E'

# 读取Excel文件
file_path = 'e41_屈原光谱数据3-fa1.xlsx'
df = pd.read_excel(file_path, sheet_name='原始光谱')
x_data = df['Chl'].values
y_data = df['Cd'].values

# 拟合多项式模型
params, r_squared = fit_function(polynomial, x_data, y_data)
# 设置全局字体
font = FontProperties(family='Times New Roman', size=16, weight='bold')
font2 = FontProperties(family='Times New Roman', size=10, weight='bold')

# 设置散点的颜色
colors_by_cd = [assign_color(cd) for cd in y_data]

# 绘制图形
plt.figure(figsize=(7, 6))
plt.grid(False)

# 绘制原始数据点
for x, y, color in zip(x_data, y_data, colors_by_cd):
    plt.scatter(x, y, c=color, s=100)

# 绘制最佳拟合线
x_fit = np.linspace(min(x_data), max(x_data), 500)
y_fit = polynomial(x_fit, *params)
plt.plot(x_fit, y_fit, label=f"Polynomial fit: $y = {params[0]:.4f}x^2 + {params[1]:.4f}x + {params[2]:.4f}$\n$R^2 = {r_squared:.4f}$", color='black')

# 设置标题和标签
plt.title('Relationship between Chl and Cd', fontsize=16, fontproperties=font)
plt.xlabel('Measured Chlorophyll content of rice (μg/cm2)', fontsize=17, fontweight='bold', fontproperties=font)
plt.ylabel('Measured Cadmium content in soil (mg/kg)', fontsize=17, fontweight='bold', fontproperties=font)

# 设置图例
legend = plt.legend(prop=font2)
for text in legend.get_texts():
    text.set_color("black")
legend.get_frame().set_edgecolor('black')
# 加粗整图外框
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.6)  # 设置外框线宽为2
font_ticks = FontProperties(family='Times New Roman', size=16, weight='bold')
plt.xticks(fontproperties=font_ticks, color='black', weight='bold')
plt.yticks(fontproperties=font_ticks, color='black', weight='bold')
# 保存图像
plt.savefig('chl_cd_relationship_plot.png', dpi=600, bbox_inches='tight')
plt.show()
