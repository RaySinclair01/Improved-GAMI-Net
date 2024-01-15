
# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
# 定义用于多项式回归的函数
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# 读取Excel文件
file_path = 'e41_Hyperspectral_data.xlsx'
df = pd.read_excel(file_path, sheet_name='原始光谱')

# 转换列名为字符串
df.columns = df.columns.astype(str)

# 计算指数值 Sum_Dr1A
indices_df = pd.DataFrame()
indices_df['Sum_Dr1A'] = df.loc[:, '625':'795'].diff(axis=1).sum(axis=1)
# 将指数值保留为三位小数
indices_df['Sum_Dr1A'] = indices_df['Sum_Dr1A'].round(3)
# 准备数据
X = indices_df['Sum_Dr1A'].values.reshape(-1, 1)
y = df['Chl'].values

# 使用2阶多项式回归进行拟合
degree = 2
model = PolynomialRegression(degree)
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# 获取2阶多项式的系数
poly_coef = model.named_steps['linearregression'].coef_
poly_intercept = model.named_steps['linearregression'].intercept_

# 创建多项式公式字符串，保留两位小数
poly_formula = f"f(x) = {poly_intercept:.2f} + {poly_coef[1]:.2f}x + {poly_coef[2]:.2f}x^2"

# 创建图
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
font = FontProperties(family='Times New Roman', size=18, weight='bold')

# ---------- 左边的子图 ----------
axs[0].scatter(X, y, c='#ff7f0e', s=100, label='Data points')
X_uniform = np.linspace(np.min(X), np.max(X), 500).reshape(-1, 1)
y_fit = model.predict(X_uniform)
axs[0].plot(X_uniform, y_fit, color='red', linestyle='-', linewidth=2, label=f"{poly_formula}\n$R^2 = {r2:.4f}$")
axs[0].set_xlabel('Sum_Dr1A index values', fontweight='bold', fontsize=20, fontproperties=font)
axs[0].set_ylabel('Measured Chl content values', fontweight='bold', fontsize=20, fontproperties=font)
axs[0].legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False, prop=font)

# 计算横轴的最小值和最大值
x_min, x_max = np.min(X), np.max(X)

# 生成自定义的横轴刻度标签
custom_ticks = np.linspace(x_min, x_max, num=10)  # 这里的 10 是自定义刻度数量
custom_ticks = np.round(custom_ticks, 2)  # 保留两位小数

# 设置自定义的横轴刻度和标签
axs[0].set_xticks(custom_ticks)
axs[0].set_xticklabels(custom_ticks)

# ---------- 右边的子图 ----------
axs[1].plot(y, y, color='black', linestyle='-', linewidth=3, label='$y=x$')
axs[1].scatter(y_pred, y, c='#ff7f0e', s=100, label='Data points')

# 使用线性回归拟合预测值和实际值
reg = LinearRegression().fit(y_pred.reshape(-1, 1), y)
y_fit_right = reg.predict(y_pred.reshape(-1, 1))
slope = reg.coef_
intercept = reg.intercept_
r2_value = r2_score(y, y_fit_right)

# 创建线性回归公式字符串，保留两位小数
linear_formula = f"$y = {slope[0]:.2f}x + {intercept:.2f}$\n$R^2 = {r2_value:.4f}$"

# 使用均匀的数据点绘制回归拟合直线（红色虚线）
y_pred_uniform = np.linspace(np.min(y_pred), np.max(y_pred), 500).reshape(-1, 1)
y_fit_uniform_right = reg.predict(y_pred_uniform)
axs[1].plot(y_pred_uniform, y_fit_uniform_right, color='red', linestyle=(0, (5, 5)), linewidth=2, label=linear_formula)

# 坐标轴标签和图例
axs[1].set_xlabel('Predicted Chl content values', fontweight='bold', fontsize=20, fontproperties=font)
axs[1].set_ylabel('Measured Chl content values', fontweight='bold', fontsize=20, fontproperties=font)
axs[1].legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False, prop=font)

# 加粗整图外框
for ax in axs:
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    font_ticks = FontProperties(family='Times New Roman', size=16, weight='bold')
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font_ticks, color='black', weight='bold')
    ax.set_yticklabels(ax.get_yticks(), fontproperties=font_ticks, color='black', weight='bold')

# 保存图像
plot_path_final = 'spectral_analysis_plot_final_2_decimal.png'
plt.savefig(plot_path_final, dpi=600, bbox_inches='tight')
plt.show()
