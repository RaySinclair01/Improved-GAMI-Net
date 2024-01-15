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

from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import global_visualize_wo_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_regularization
from gaminet.utils import plot_trajectory


# In[3]:


task_type = "Classification"

data = pd.read_csv("./bank.csv", sep=";")
meta_info = json.load(open("./data_types.json"))
data['month'] = data['month'].replace('jan', 1).replace('feb', 2).replace('mar', 3).replace('apr', 4).\
                              replace('may', 5).replace('jun', 6).replace('jul', 7).replace('aug', 8).\
                              replace('sep', 9).replace('oct', 10).replace('nov', 11).replace('dec', 12)
x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
xx = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
for i, (key, item) in enumerate(meta_info.items()):
    if item['type'] == 'target':
        enc = OrdinalEncoder()
        enc.fit(y)
        y = enc.transform(y)
        meta_info[key]['values'] = enc.categories_[0].tolist()
    elif item['type'] == 'categorical':
        enc = OrdinalEncoder()
        xx[:,[i]] = enc.fit_transform(x[:,[i]])
        meta_info[key]['values'] = []
        for item in enc.categories_[0].tolist():
            try:
                if item == int(item):
                    meta_info[key]['values'].append(str(int(item)))
                else:
                    meta_info[key]['values'].append(str(item))
            except ValueError:
                meta_info[key]['values'].append(str(item))
    else:
        sx = MinMaxScaler((0, 1))
        xx[:,[i]] = sx.fit_transform(x[:,[i]])
        meta_info[key]['scaler'] = sx
train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=0.2, random_state=0)


# In[4]:


def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def auc(label, pred, scaler=None):
    return roc_auc_score(label, pred)

get_metric = metric_wrapper(auc, None)


# In[5]:


folder = "./results/"
if not os.path.exists(folder):
    os.makedirs(folder)

model_bank = GAMINet(meta_info=meta_info, interact_num=20,
            interact_arch=[40] * 1, subnet_arch=[40] * 2,
            batch_size=200, task_type=task_type, activation_func=tf.nn.relu,
            main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
            lr_bp=[0.001, 0.001, 0.001], early_stop_thres=[50, 50, 50],
            heredity=True, loss_threshold=0.01, reg_clarity=0.1,
            mono_increasing_list=[11],
            mono_decreasing_list=[0],
            lattice_size=10,
            verbose=True, val_ratio=0.2, random_state=0)
model_bank.fit(train_x, train_y, sample_weight=np.random.uniform(0, 1, size=(train_x.shape[0], 1)))
data_dict_logs = model_bank.summary_logs(save_dict=False)
plot_trajectory(data_dict_logs, folder=folder, name="bank_traj", save_png=True, save_eps=True)
plot_regularization(data_dict_logs, folder=folder, name="bank_regu", save_png=True, save_eps=True)


# Global Interpretation

# In[6]:


data_dict_global = model_bank.global_explain(save_dict=True, folder=folder, name="bank_global")
global_visualize_density(data_dict_global, folder=folder, name="bank_global",
                         main_effect_num=8, interaction_num=4, cols_per_row=4, save_png=True, save_eps=True)


# Feature Importance

# In[7]:


feature_importance_visualize(data_dict_global)


# Interpret the prediction of a test sample

# In[8]:


data_dict_local = model_bank.local_explain(test_x[[0]], test_y[[0]], save_dict=False)
local_visualize(data_dict_local[0], save_png=False)


# ## Model save and load

# In[9]:


model_bank.save(folder="./", name="model01_saved")


# In[10]:


## The reloaded model should not be refit again
modelnew = GAMINet(meta_info={})
modelnew.load(folder="./", name="model01_saved")


# In[11]:


pred_train = modelnew.predict(train_x)
pred_test = modelnew.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(train_y, pred_train),5),
                      np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)


# In[12]:




