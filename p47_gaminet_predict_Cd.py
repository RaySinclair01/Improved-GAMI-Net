import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error,make_scorer,r2_score
from gaminet import GAMINet
import matplotlib.pyplot as plt
from gaminet.utils import local_visualize,global_visualize_density,global_visualize_wo_density,feature_importance_visualize,plot_regularization,plot_trajectory
from sklearn.linear_model import LinearRegression
# In[3]:读入数据
data = pd.read_csv("./GAMI-NET_Chl.csv", sep=",")
meta_info = json.load(open("./GAMI-NET_Chl_data_types.json"))
x1, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values

# In[4]:数据预处理
# 目标变量归一化
scaler_y = MinMaxScaler((0, 1))
y = scaler_y.fit_transform(y)
# 特征变量归一化
xx = np.zeros((x1.shape[0], x1.shape[1]), dtype=np.float32)
for i, (key, item) in enumerate(meta_info.items()):
    if item['type'] == 'target':
        # 已经进行了归一化，无需其他操作
        pass
    else:  # 假设所有其他特征都是 'continuous'
        sx = MinMaxScaler((0, 1))
        xx[:, [i]] = sx.fit_transform(x1[:, [i]])
        meta_info[key]['scaler'] = sx
# 训练集和测试集的设定
test_x=x1
test_y=y

# In[20]:模型加载

## The reloaded model should not be refit again
modelnew = GAMINet(meta_info={})
modelnew.load(folder="./", name="model_Chl29_saved")
# In[5]:定义模型评估指标
def mse(label, pred, scaler=None):
    return mean_squared_error(label, pred)
get_metric = mse  # 直接使用MSE作为评估指标

# In[21]:测试集上的预测效果及其评估指标mse
pred_test = modelnew.predict(test_x)
# 逆归一化预测结果
pred_test_inv = scaler_y.inverse_transform(pred_test)

# 评估指标也需要用原始数据尺度来计算
test_y_inv = scaler_y.inverse_transform(test_y)
# 计算MSE
mse_value = mean_squared_error(test_y_inv, pred_test_inv)
print(mse_value)

# 设置全局字体为 "Times New Roman"
plt.rc('font', family='Times New Roman')

# 计算回归拟合直线
reg = LinearRegression().fit(test_y_inv.reshape(-1, 1), pred_test_inv)
pred_fit = reg.predict(test_y_inv.reshape(-1, 1))
# 创建图
plt.figure(figsize=(8, 8))
min_value = min(test_y_inv)[0]
max_value = max(test_y_inv)[0]
# 绘制1:1线（黑色实线）
plt.plot([min_value, max_value], [min_value, max_value],
         color='black', linestyle='-', linewidth=2, label='1:1 line')

# 绘制回归拟合直线（红色虚线）
plt.plot(test_y_inv, pred_fit, color='red', linestyle='--', linewidth=2, label='Fit line')

# 绘制散点（颜色为 #ff7f0e，点更大）
plt.scatter(test_y_inv, pred_test_inv, c='#ff7f0e', s=100, label='Data points')  # s 参数设置点的大小


min_value = min(test_y_inv)[0]
max_value = max(test_y_inv)[0]
# 添加置信区间（假设为±0.5）
plt.fill_between([min_value, max_value],
                 [min_value - 0.5, max_value - 0.5],
                 [min_value + 0.5, max_value + 0.5],
                 color='grey', alpha=0.5, label='Confidence interval')


# 添加标签和图例（字体加粗，字体更大）
plt.xlabel('test_y_inv values', fontweight='bold', fontsize=14)
plt.ylabel('pred_test_inv values', fontweight='bold', fontsize=14)
plt.legend(fontsize=12, loc='upper left')


# 显示图
plt.show()