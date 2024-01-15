import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
# 加载数据
file_path = '屈原光谱数据3-fa1.xlsx'
sheet_data = pd.read_excel(file_path, sheet_name=0)
groups = ['clean', 'low', 'medium', 'high']
spectra_data = {group: sheet_data[sheet_data['group'] == group].iloc[:, 4:] for group in groups}


# 定义去除包络线的函数
def remove_envelope_hilbert(spectrum):
    analytic_signal = hilbert(spectrum)
    envelope = np.abs(analytic_signal)
    envelope_removed_spectrum = spectrum - envelope
    return envelope_removed_spectrum


# 对每组的每个样本进行包络线去除操作并求平均
envelope_removed_spectra_data_hilbert = {group: data.apply(remove_envelope_hilbert, axis=1) for group, data in spectra_data.items()}
average_envelope_removed_spectra = {group: data.mean() for group, data in envelope_removed_spectra_data_hilbert.items()}


# 定义取倒数对数的函数
def reciprocal_log(spectrum):
    return np.log(1 / spectrum.astype(float)) # 修改此处

# 对每组的每个样本进行取倒数对数操作并求平均
reciprocal_log_spectra_data = {group: data.apply(reciprocal_log, axis=1) for group, data in spectra_data.items()} # 修改函数名
average_reciprocal_log_spectra = {group: data.mean() for group, data in reciprocal_log_spectra_data.items()} # 修改函数名


# 定义绘图函数
def plot_comparison(group1, group2, color1, color2, ax, title_suffix=''):
    ax.plot(average_envelope_removed_spectra[group1], color=color1, label=f'{group1} group')
    ax.plot(average_envelope_removed_spectra[group2], color=color2, label=f'{group2} group')
    ax.fill_between(range(325, 1076), average_envelope_removed_spectra[group1], average_envelope_removed_spectra[group2], facecolor='gray', alpha=0.5)
    ax.set_title(f'Comparison of {group1} and {group2} groups {title_suffix}')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Reflectance')
    ax.legend()

# 定义绘图函数，用于取倒数对数的图
def plot_inverse_log_comparison(group1, group2, color1, color2, ax):
    ax.plot(average_reciprocal_log_spectra[group1], color=color1, label=f'{group1} group')
    ax.plot(average_reciprocal_log_spectra[group2], color=color2, label=f'{group2} group')
    ax.fill_between(range(325, 1076), average_reciprocal_log_spectra[group1], average_reciprocal_log_spectra[group2], facecolor='gray', alpha=0.5)
    ax.set_title(f'Comparison of {group1} and {group2} groups (Reciprocal Log)') # 更新标题
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Reciprocal Log of Reflectance') # 更新纵轴标签
    ax.legend()

# 获取下方三个图（取倒数对数的）的纵轴范围
y_min = min(average_reciprocal_log_spectra[group].min() for group in groups)
y_max = max(average_reciprocal_log_spectra[group].max() for group in groups)+0.2

# 创建子图并绘制，设置下方三个图的纵轴范围
fig, axes = plt.subplots(2, 3, figsize=[16, 8])
plot_comparison('clean', 'low', 'turquoise', 'orange', axes[0, 0], '(Envelope Removed)')
plot_comparison('clean', 'medium', 'turquoise', 'red', axes[0, 1], '(Envelope Removed)')
plot_comparison('clean', 'high', 'turquoise', 'darkred', axes[0, 2], '(Envelope Removed)')
plot_inverse_log_comparison('clean', 'low', 'turquoise', 'orange', axes[1, 0])
plot_inverse_log_comparison('clean', 'medium', 'turquoise', 'red', axes[1, 1])
plot_inverse_log_comparison('clean', 'high', 'turquoise', 'darkred', axes[1, 2])
axes[1, 0].set_ylim(y_min, y_max) # 设置纵轴范围
axes[1, 1].set_ylim(y_min, y_max) # 设置纵轴范围
axes[1, 2].set_ylim(y_min, y_max) # 设置纵轴范围
plt.tight_layout()
plt.show()

