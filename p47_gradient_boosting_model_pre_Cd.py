
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
# 读入数据
data = pd.read_excel('e12_GAMI-NET_Cd_toPre.xlsx', skiprows=[0])
x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values

# 数据预处理
# 归一化目标变量
scaler_y = MinMaxScaler((0, 1))
y = scaler_y.fit_transform(y)

# 归一化特征变量
train_x = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
sx = MinMaxScaler((0, 1))
train_x = sx.fit_transform(x)
train_y = y

# 定义梯度提升树模型并训练
gb_model = GradientBoostingRegressor(n_estimators=60, max_depth=2,random_state=42)
gb_model.fit(train_x, train_y.ravel())

# 预测
pred_train_gb = gb_model.predict(train_x).reshape(-1, 1)

# 反归一化
pred_train_gb_inv = scaler_y.inverse_transform(pred_train_gb)
train_y_inv = scaler_y.inverse_transform(train_y)
# 计算回归拟合直线
reg_gb = LinearRegression().fit(train_y_inv, pred_train_gb_inv)
pred_fit_gb = reg_gb.predict(train_y_inv)

# 线性方程参数
slope_gb = reg_gb.coef_[0]
intercept_gb = reg_gb.intercept_
r2_value_gb = r2_score(train_y_inv, pred_train_gb_inv)

# 创建方程和R^2文本
equation_text_gb = f"y = {slope_gb[0]:.4f}x + {intercept_gb[0]:.4f}\n$R^2$ = {r2_value_gb:.4f}"

# 设置全局字体
font = FontProperties(family='Times New Roman', size=18, weight='bold')

# 创建图
plt.figure(figsize=(8, 8))

# 绘制1:1线（黑色粗实线）
plt.plot(train_y_inv, train_y_inv, color='black', linestyle='-', linewidth=3, label='$y=x$')

# 生成均匀的数据点用于绘制回归拟合直线
train_y_inv_uniform_gb = np.linspace(np.min(train_y_inv), np.max(train_y_inv), 500).reshape(-1, 1)
pred_fit_uniform_gb = reg_gb.predict(train_y_inv_uniform_gb)

# 使用均匀的数据点绘制回归拟合直线（红色虚线）
plt.plot(train_y_inv_uniform_gb, pred_fit_uniform_gb, color='red', linestyle=(0, (5, 5)), linewidth=2, label=f"$y = {slope_gb[0]:.4f}x + {intercept_gb[0]:.4f}$\n$R^2 = {r2_value_gb:.4f}$")

# 添加置信区间（淡蓝色区域）
min_val_gb = np.min(train_y_inv)
max_val_gb = np.max(train_y_inv)
plt.fill_between([min_val_gb, max_val_gb], [min_val_gb - 0.4, max_val_gb - 0.4], [min_val_gb + 0.4, max_val_gb + 0.4], color='lightblue', alpha=0.5, label='Confidence interval')
# 绘制散点图
plt.scatter(train_y_inv, pred_train_gb_inv, c='#ff7f0e', s=100, label='Data points')


# 坐标轴标签和图例
plt.xlabel(r'Measured Cd content values', fontweight='bold', fontsize=20, fontproperties=font)
plt.ylabel(r'Predicted Cd content values', fontweight='bold', fontsize=20, fontproperties=font)
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False, prop=font)

# 加粗整图外框
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)  # 设置外框线宽为2
font_ticks = FontProperties(family='Times New Roman', size=16, weight='bold')
plt.xticks(fontproperties=font_ticks, color='black', weight='bold')
plt.yticks(fontproperties=font_ticks, color='black', weight='bold')
# 保存图像
plot_path_gb = 'p47_xgboosttree_pre_Cd.png'
plt.savefig(plot_path_gb, dpi=600, bbox_inches='tight')
plt.show()


