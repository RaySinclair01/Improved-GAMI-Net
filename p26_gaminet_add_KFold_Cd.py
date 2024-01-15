#!/usr/bin/env python
# coding: utf-8

# In[1]:
  #  import sys
  #  PACKAGE_PARENT = '..'
  #  sys.path.append(PACKAGE_PARENT)

# In[2]:导入模块
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error,make_scorer
from gaminet import GAMINet
from gaminet.utils import local_visualize,global_visualize_density,global_visualize_wo_density,feature_importance_visualize,plot_regularization,plot_trajectory

# In[3]:读入数据
data = pd.read_csv("./GAMI-NET_Cd.csv", sep=",")
meta_info = json.load(open("./GAMI-NET_Cd_data_types.json"))
x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values

# In[4]:数据预处理
# 目标变量归一化
scaler_y = MinMaxScaler((0, 1))
y = scaler_y.fit_transform(y)
# 特征变量归一化
xx = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
for i, (key, item) in enumerate(meta_info.items()):
    if item['type'] == 'target':
        # 已经进行了归一化，无需其他操作
        pass
    else:  # 假设所有其他特征都是 'continuous'
        sx = MinMaxScaler((0, 1))
        xx[:, [i]] = sx.fit_transform(x[:, [i]])
        meta_info[key]['scaler'] = sx
# 训练集和测试集的拆分4:1
train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=0.2, random_state=0)

# In[5]:定义模型评估指标
def mse(label, pred, scaler=None):
    return mean_squared_error(label, pred)
get_metric = mse  # 直接使用MSE作为评估指标
# Convert custom metric to a scorer function
scorer = make_scorer(get_metric, greater_is_better=False)

# In[6]:初始化KFold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# In[7]:设置超参数网格
param_grid = {
    'interact_num': [5],
    'interact_arch': [[10] * 1, [20] * 1],
    'subnet_arch': [[10] *1 ,[20] * 1],
    'batch_size': [5],
    'main_effect_epochs': [80],
    'interaction_epochs': [80],
    'tuning_epochs': [20],
    'lr_bp': [[0.0001, 0.0001, 0.0001], [0.0005, 0.0005, 0.0005]],
    'early_stop_thres': [[10, 10, 10]],
    'reg_clarity': [0.3],
    'lattice_size': [2],
    'verbose': [True],  # 打印训练过程
    'val_ratio': [0.2],  # 使用LOO这个参数无效
    'random_state': [0],  # 固定随机种子
    'heredity': [True],#使用遗传算法
    'loss_threshold': [0.01]
}

# In[8]:初始化GAMINET模型
task_type = "Regression"
folder = "./results-Cd/"
if not os.path.exists(folder):
    os.makedirs(folder)
model_Cd = GAMINet(meta_info=meta_info, interact_num=5,
            interact_arch=[10] * 1, subnet_arch=[10] * 1,
            batch_size=5, task_type=task_type, activation_func=tf.nn.relu,
            main_effect_epochs=50, interaction_epochs=50, tuning_epochs=20,
            lr_bp=[0.0005, 0.0005, 0.0005], early_stop_thres=[10, 10, 10],
            heredity=True, loss_threshold=0.01, reg_clarity=0.3,
            #mono_increasing_list=[11],
            #mono_decreasing_list=[0],
            lattice_size=2,
            verbose=True, val_ratio=0.2, random_state=0)

# In[9]:初始化GridSearchCV 和 LeaveOneOut cross-validator ,并设置使用40个进程
grid_search = GridSearchCV(model_Cd, param_grid=param_grid, scoring=scorer, cv=kfold, n_jobs=1, verbose=1)
# In[10]:训练模型 perform grid search and hyperparameter tuning
grid_best_model = grid_search.fit(train_x, train_y)

# In[11]:输出最优超参数和指标
print("Best parameters found:",grid_best_model.best_params_)
print("Best score found:",grid_best_model.best_score_)

# In[12]:输出最优模型在测试集上的预测效果及其评估指标mse
pred_train = grid_best_model.predict(train_x)
pred_test = grid_best_model.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(train_y, pred_train),5),
                        np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)
# In[13]:输出最优模型并画图
print("Best model found:",grid_best_model.best_estimator_)
model_Cd2=grid_best_model.best_estimator_
#model_Cd.fit(train_x, train_y, sample_weight=np.random.uniform(0, 1, size=(train_x.shape[0], 1)))
data_dict_logs = model_Cd2.summary_logs(save_dict=False)
plot_trajectory(data_dict_logs, folder=folder, name="Cd_traj", save_png=True, save_eps=True)
plot_regularization(data_dict_logs, folder=folder, name="Cd_regu", save_png=True, save_eps=True)


# Global Interpretation

# In[14]:全局解释可视化


data_dict_global = model_Cd2.global_explain(save_dict=True, folder=folder, name="Cd_global")
global_visualize_density(data_dict_global, folder=folder, name="Cd_global",
                         main_effect_num=8, interaction_num=4, cols_per_row=4, save_png=True, save_eps=True)


# Feature Importance

# In[15]:特征重要性可视化


feature_importance_visualize(data_dict_global)


# Interpret the prediction of a test sample

# In[16]:局部解释可视化



data_dict_local = model_Cd2.local_explain(test_x[[0]], test_y[[0]], save_dict=False)
local_visualize(data_dict_local[0], save_png=True)


# ## Model save and load

# In[17]:模型保存


model_Cd2.save(folder="./", name="model_Cd2_saved")


# In[18]:模型加载


## The reloaded model should not be refit again
modelnew = GAMINet(meta_info={})
modelnew.load(folder="./", name="model_Cd2_saved")


# In[19]:测试集上的预测效果及其评估指标mse




pred_train = modelnew.predict(train_x)
pred_test = modelnew.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(train_y, pred_train),5),
                      np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)






