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
from matplotlib import gridspec
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error,make_scorer,r2_score
from gaminet import GAMINet
from gaminet.utils import local_visualize,global_visualize_density,global_visualize_wo_density,feature_importance_visualize,plot_regularization,plot_trajectory
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
# In[3]:读入数据
data = pd.read_csv("./GAMI-NET_Chl.csv", sep=",")
meta_info = json.load(open("./GAMI-NET_Chl_data_types.json"))
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

# In[8]:初始化GAMINET模型
task_type = "Regression"
folder = "./results-Chl28-1/"
if not os.path.exists(folder):
    os.makedirs(folder)
grid_best_model = GAMINet(meta_info=meta_info, interact_num=5,
            interact_arch=[10] * 1, subnet_arch=[10] * 1,
            batch_size=5, task_type=task_type, activation_func=tf.nn.relu,
            main_effect_epochs=50, interaction_epochs=50, tuning_epochs=20,
            lr_bp=[0.0005, 0.0005, 0.0005], early_stop_thres=[10, 10, 10],
            heredity=True, loss_threshold=0.01, reg_clarity=0.3,
            #mono_increasing_list=[11],
            #mono_decreasing_list=[0],
            lattice_size=2,
            verbose=True, val_ratio=0.2, random_state=0)
grid_best_model.load(folder="./", name="model_Chl28_saved")


combined_x = np.vstack([train_x, test_x])
# 输出最优模型在训练集上的预测效果及其评估指标mse
pred_tt = grid_best_model.predict(combined_x)
pred_tt_inv = scaler_y.inverse_transform(pred_tt)
tt_inv = scaler_y.inverse_transform(y)
# 计算训练集上的 R2
r2_tt = r2_score(pred_tt_inv, tt_inv)
print("Current R2 on training set:", r2_tt)


print("Achieved R2 > 0.8 on the training set. Stop retraining.")
# 计算MSE
mse_value = mean_squared_error(tt_inv, pred_tt_inv)
print(mse_value)


# In[12]:输出最优模型在测试集上的预测效果及其评估指标mse
pred_train = grid_best_model.predict(train_x)
pred_test = grid_best_model.predict(test_x)
gaminet_stat = np.hstack([np.round(get_metric(train_y, pred_train),5),
                        np.round(get_metric(test_y, pred_test),5)])
print(gaminet_stat)

# In[14]:输出最优模型在训练集和测试集上的预测效果及其评估指标R2,RMSE,RPD
# 计算 R2
r2_train = r2_score(train_y, pred_train)
r2_test = r2_score(test_y, pred_test)

# 计算 RMSE
rmse_train = np.sqrt(mean_squared_error(train_y, pred_train))
rmse_test = np.sqrt(mean_squared_error(test_y, pred_test))

# 计算 RPD
rpd_train = np.std(train_y) / rmse_train
rpd_test = np.std(test_y) / rmse_test
# Global Interpretation
# In[15]:保存R2,RMSE,RPD到excel
# 创建一个 DataFrame 来存储这些指标
df_metrics = pd.DataFrame({
    'Metric': ['R2', 'RMSE', 'RPD'],
    'Train': [r2_train, rmse_train, rpd_train],
    'Test': [r2_test, rmse_test, rpd_test]
})
# 将 DataFrame 保存为 Excel 文件
df_metrics.to_excel("Chl_gaminet_metrics28.xlsx", index=False)


def feature_importance_visualize1(data_dict_global, folder="./results/", name="demo", save_png=False, save_eps=False):
    all_ir = []
    all_names = []

    # 遍历全局数据字典以获取重要性和名称
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)

    # 排序和颜色分配
    sorted_indices = np.argsort(all_ir)
    colors = []
    for i in sorted_indices:
        if "vs" in all_names[i]:
            colors.append("#E45C5E")
        else:
            colors.append("#004775")

    max_ids = len(all_names)
    if max_ids > 0:
        fig, ax = plt.subplots(figsize=(6, 0.2 + 0.4 * max_ids))

        # 使用指定的颜色和排序绘制横向条形图
        bars = ax.barh(np.arange(len(all_ir)), np.array(all_ir)[sorted_indices], color=colors)

        # 添加末端数字
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2.0 - 0.1, f"{width * 100:.1f}%", fontsize=12,
                    fontname='Times New Roman', fontweight='bold')  # va: vertical alignment

        # 设置字体和其他格式
        ax.set_yticks(np.arange(len(all_ir)))
        ax.set_yticklabels(np.array(all_names)[sorted_indices], fontsize=12, fontname='Times New Roman',
                           fontweight='bold')
        ax.set_ylabel("Feature Name", fontsize=12, fontname='Times New Roman', fontweight='bold')
        ax.set_title("Feature Importance", fontsize=12, fontname='Times New Roman', fontweight='bold')
        plt.yticks(fontsize=12, fontname='Times New Roman', fontweight='bold')

        # 将横轴刻度转换为百分数
        vals = ax.get_xticks()
        ax.set_xticks(vals)
        ax.set_xticklabels([f"{x * 100:.0f}%" for x in vals], fontname='Times New Roman', fontsize=12,
                           fontweight='bold', color='black')

        # 设置外框
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        plt.xlim(0, np.max(all_ir) + 0.05)
        plt.ylim(-1, len(all_names))

        # 调整刻度标签位置
        ax.get_yaxis().set_tick_params(pad=15)

        # 添加图例并设置字体和颜色
        legend_font = FontProperties(family='Times New Roman', weight='bold', size=12)
        legend = ax.legend([plt.Rectangle((0, 0), 1, 1, fc="#004775", edgecolor="none"),
                            plt.Rectangle((0, 0), 1, 1, fc="#E45C5E", edgecolor="none")],
                           ["Main Effect", "Interaction Effect"], loc='lower right', frameon=False, fontsize=12)

        for text in legend.get_texts():
            text.set_fontproperties(legend_font)

        # 微调图例位置
        legend.set_bbox_to_anchor((0.99, 0.05))

        # 保存图像
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=600)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=600)
def global_visualize_density1(data_dict_global, main_effect_num=None, interaction_num=None, cols_per_row=4,
                             save_png=False, save_eps=False, folder="./results/", name="demo"):
    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum() > 0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    if max_ids == 0:
        return

    idx = 0
    font_properties = FontProperties(family='Times New Roman', weight='bold', size=14)
    fig = plt.figure(figsize=(6 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.2, hspace=0.3)  # 调整了间距


    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1,
                                                     height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            xint = ((np.array(data_dict_global[feature_name]["density"]["names"][1:])
                     + np.array(data_dict_global[feature_name]["density"]["names"][:-1])) / 2).reshape([-1, 1]).reshape(
                [-1])
            ax2.bar(xint, data_dict_global[feature_name]["density"]["scores"], width=xint[1] - xint[0])
            ax2.get_shared_x_axes().join(ax1, ax2)
            ax2.set_yticklabels([])
            ax2.autoscale()
            fig.add_subplot(ax2)

        elif data_dict_global[feature_name]["type"] == "categorical":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx],
                                                     wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                    data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            ax2.bar(np.arange(len(data_dict_global[feature_name]["density"]["names"])),
                    data_dict_global[feature_name]["density"]["scores"])
            ax2.get_shared_x_axes().join(ax1, ax2)
            ax2.autoscale()
            ax2.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax2.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            ax2.set_yticklabels([])
            fig.add_subplot(ax2)

        idx = idx + 1
        if len(str(ax2.get_xticks())) > 60:
            ax2.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)",
                      fontproperties=font_properties)
        ax1.tick_params(axis='both', which='major', labelsize=14)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        feature_name1 = feature_name.split(" vs. ")[0]
        feature_name2 = feature_name.split(" vs. ")[1]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[idx],
                                                 wspace=0.1, hspace=0.1, height_ratios=[6, 1],
                                                 width_ratios=[0.6, 3, 0.15, 0.2])
        ax_main = plt.Subplot(fig, inner[1])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                                       aspect="auto", extent=axis_extent)
        ax_main.set_xticklabels([])
        ax_main.set_yticklabels([])
        ax_main.set_title(
            feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)",
            fontproperties=font_properties)
        ax_main.tick_params(axis='both', which='major', labelsize=14)
        fig.add_subplot(ax_main)

        ax_bottom = plt.Subplot(fig, inner[5])
        if data_dict_global[feature_name]["xtype"] == "categorical":
            xint = np.arange(len(data_dict_global[feature_name1]["density"]["names"]))
            ax_bottom.bar(xint, data_dict_global[feature_name1]["density"]["scores"])
            ax_bottom.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_bottom.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        else:
            xint = ((np.array(data_dict_global[feature_name1]["density"]["names"][1:])
                     + np.array(data_dict_global[feature_name1]["density"]["names"][:-1])) / 2).reshape([-1])
            ax_bottom.bar(xint, data_dict_global[feature_name1]["density"]["scores"], width=xint[1] - xint[0])
        ax_bottom.set_yticklabels([])
        ax_bottom.set_xlim([axis_extent[0], axis_extent[1]])
        ax_bottom.get_shared_x_axes().join(ax_bottom, ax_main)
        ax_bottom.autoscale()
        fig.add_subplot(ax_bottom)
        if len(str(ax_bottom.get_xticks())) > 60:
            ax_bottom.xaxis.set_tick_params(rotation=20)

        ax_left = plt.Subplot(fig, inner[0])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            xint = np.arange(len(data_dict_global[feature_name2]["density"]["names"]))
            ax_left.barh(xint, data_dict_global[feature_name2]["density"]["scores"])
            ax_left.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_left.set_yticklabels(data_dict_global[feature_name]["input2_labels"])
        else:
            xint = ((np.array(data_dict_global[feature_name2]["density"]["names"][1:])
                     + np.array(data_dict_global[feature_name2]["density"]["names"][:-1])) / 2).reshape([-1])
            ax_left.barh(xint, data_dict_global[feature_name2]["density"]["scores"], height=xint[1] - xint[0])
        ax_left.set_xticklabels([])
        ax_left.set_ylim([axis_extent[2], axis_extent[3]])
        ax_left.get_shared_y_axes().join(ax_left, ax_main)
        ax_left.autoscale()
        fig.add_subplot(ax_left)

        ax_colorbar = plt.Subplot(fig, inner[2])
        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"])
                                                - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        fig.colorbar(interact_plot, cax=ax_colorbar, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)
        fig.add_subplot(ax_colorbar)
        idx = idx + 1

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=600)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=600)

# In[16]:全局解释可视化


data_dict_global = grid_best_model.global_explain(save_dict=True, folder=folder, name="Chl_global")
global_visualize_density1(data_dict_global, folder=folder, name="Chl_global",
                         main_effect_num=8, interaction_num=4, cols_per_row=4, save_png=True, save_eps=True)


# Feature Importance

# In[17]:特征重要性可视化


feature_importance_visualize1(data_dict_global, folder=folder, name="Chl_feature_importance",save_png=True)


# Interpret the prediction of a test sample

# In[18]:局部解释可视化



data_dict_local = grid_best_model.local_explain(test_x[[0]], test_y[[0]], save_dict=False)
local_visualize(data_dict_local[0], folder=folder, name="Chl_local_importance",save_png=True)









