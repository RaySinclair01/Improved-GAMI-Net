#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

PACKAGE_PARENT = '..'
sys.path.append(PACKAGE_PARENT)


# In[2]:


import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import global_visualize_wo_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_regularization
from gaminet.utils import plot_trajectory


# In[3]:


task_type = "Regression"

data = pd.read_csv("./GAMI-NET_Chl.csv", sep=",")
meta_info = json.load(open("./GAMI-NET_Chl_data_types.json"))
x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
xx = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
# 目标变量归一化
scaler_y = MinMaxScaler((0, 1))
y = scaler_y.fit_transform(y)

for i, (key, item) in enumerate(meta_info.items()):
    if item['type'] == 'target':
        # 已经进行了归一化，无需其他操作
        pass
    else:  # 假设所有其他特征都是 'continuous'
        sx = MinMaxScaler((0, 1))
        xx[:, [i]] = sx.fit_transform(x[:, [i]])
        meta_info[key]['scaler'] = sx
train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=0.2, random_state=0)


# In[4]:


def mse(label, pred, scaler=None):
    return mean_squared_error(label, pred)

get_metric = mse  # 直接使用MSE作为评估指标


# In[5]:


folder = "./results-Chl/"
if not os.path.exists(folder):
    os.makedirs(folder)

model_Chl = GAMINet(meta_info=meta_info, interact_num=20,
            interact_arch=[20] * 1, subnet_arch=[20] * 2,
            batch_size=10, task_type=task_type, activation_func=tf.nn.relu,
            main_effect_epochs=100, interaction_epochs=100, tuning_epochs=50,
            lr_bp=[0.0005, 0.0005, 0.0005], early_stop_thres=[10, 10, 10],
            heredity=True, loss_threshold=0.01, reg_clarity=0.3,
            #mono_increasing_list=[11],
            #mono_decreasing_list=[0],
            lattice_size=3,
            verbose=True, val_ratio=0.2, random_state=0)
model_Chl.fit(train_x, train_y, sample_weight=np.random.uniform(0, 1, size=(train_x.shape[0], 1)))
data_dict_logs = model_Chl.summary_logs(save_dict=False)
plot_trajectory(data_dict_logs, folder=folder, name="Chl_traj", save_png=True, save_eps=True)
plot_regularization(data_dict_logs, folder=folder, name="Chl_regu", save_png=True, save_eps=True)


# Global Interpretation

# In[6]:


data_dict_global = model_Chl.global_explain(save_dict=True, folder=folder, name="Chl_global")
global_visualize_density(data_dict_global, folder=folder, name="Chl_global",
                         main_effect_num=8, interaction_num=4, cols_per_row=4, save_png=True, save_eps=True)


# Feature Importance

# In[7]:


feature_importance_visualize(data_dict_global)


# Interpret the prediction of a test sample

# In[8]:


data_dict_local = model_Chl.local_explain(test_x[[0]], test_y[[0]], save_dict=False)
local_visualize(data_dict_local[0], save_png=True)


# ## Model save and load

# In[9]:


model_Chl.save(folder="./", name="model_Chl_saved")


# In[10]:


## The reloaded model should not be refit again
modelnew = GAMINet(meta_info={})
modelnew.load(folder="./", name="model_Chl_saved")


# In[11]:


pred_train = modelnew.predict(train_x)
pred_test = modelnew.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(train_y, pred_train),5),
                      np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)


# In[12]:




