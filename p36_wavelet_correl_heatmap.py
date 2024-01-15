# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

# 读取数据
file_path = "Hyperspectral_data.xlsx"
data = pd.read_excel(file_path, sheet_name=0)

# 选择波长范围
wavelengths_range = range(325, 1076)
spectra_data = data.iloc[:, 4:]

# 定义小波变换函数和其他函数，这些函数在 原始脚本中
def wavelet_transform(data, wavelet_name, level):
    coeffs = pywt.wavedec(data, wavelet_name, level=level)
    return coeffs

def calculate_metrics(subwave):
    metrics = {
        'Mean': np.mean(subwave),
        'Energy': np.sum(np.square(subwave)),
        'Variance': np.var(subwave),
        'Kurtosis': pd.Series(subwave).kurtosis(),
        'Skewness': pd.Series(subwave).skew(),
        'Max': np.max(subwave),
        'Min': np.min(subwave),
        '25th Percentile': np.percentile(subwave, 25),
        '50th Percentile': np.percentile(subwave, 50),
        '75th Percentile': np.percentile(subwave, 75),
        'Range': np.max(subwave) - np.min(subwave)
    }
    return metrics

def sort_correlations(correlations, subwavelet_names, metric_names):
    sorted_correlations = []
    sorted_labels = []
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            sorted_correlations.append(correlations[i, j])
            sorted_labels.append(f"{subwavelet_names[i]}-{metric_names[j]}")
    sorted_correlations, sorted_labels = zip(*sorted((x, y) for x, y in zip(sorted_correlations, sorted_labels)))
    return np.array(sorted_correlations)[::-1], np.array(sorted_labels)[::-1]

# 定义子波名称和指标名称
subwavelet_names = [f"D{d}" for d in range(1, 11)]
metric_names = ['Mean', 'Energy', 'Variance', 'Kurtosis', 'Skewness', 'Max', 'Min', '25th Percentile', '50th Percentile', '75th Percentile', 'Range']

# 三维矩阵初始化
num_samples = spectra_data.shape[0]
num_subwaves = len(subwavelet_names)
num_metrics = len(metric_names)
three_dim_matrix = np.zeros((num_samples, num_subwaves, num_metrics))

# 获取信号长度
signal_length = spectra_data.shape[1]

# 计算可以应用的最大小波变换级别
max_level = pywt.dwt_max_level(signal_length, 'db1')

# 进行小波变换并计算指标
for sample_idx in range(num_samples):
    sample_data = spectra_data.iloc[sample_idx].values
    coeffs = wavelet_transform(sample_data, 'db1', level=max_level)

    # 确保子波数量与三维矩阵的大小匹配
    for subwave_idx, subwave in enumerate(coeffs[:num_subwaves]):
        metrics = calculate_metrics(subwave)
        for metric_idx, metric_name in enumerate(metric_names):
            three_dim_matrix[sample_idx, subwave_idx, metric_idx] = metrics[metric_name]

# 计算与Cd和Chl的相关性
correlations_cd = np.zeros((num_subwaves, num_metrics))
correlations_chl = np.zeros((num_subwaves, num_metrics))

for subwave_idx in range(num_subwaves):
    for metric_idx in range(num_metrics):
        metric_values = three_dim_matrix[:, subwave_idx, metric_idx]
        correlations_cd[subwave_idx, metric_idx] = abs(np.corrcoef(metric_values, data['Cd'])[0, 1])
        correlations_chl[subwave_idx, metric_idx] = abs(np.corrcoef(metric_values, data['Chl'])[0, 1])

# 替换NaN值为0
correlations_cd[np.isnan(correlations_cd)] = 0
correlations_chl[np.isnan(correlations_chl)] = 0

# 字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 18

# 绘制图形
fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # 只绘制上面两个图
# 重新绘制Chl相关性热图，并添加格子
bx = sns.heatmap(correlations_chl, cmap=sns.color_palette("vlag", as_cmap=True), vmin=0, vmax=1, ax=axes[0],
            xticklabels=metric_names, yticklabels=subwavelet_names, cbar_kws={'label': 'Absolute Correlation'},
            linewidths=.5, linecolor='white')  # 添加灰色格子
cbar = bx.collections[0].colorbar
cbar.set_label('Absolute Correlation', fontsize=20, fontweight='bold')
axes[0].set_title("Correlations with Chl", fontsize=24, fontweight='bold')
axes[0].set_xlabel("Metrics", fontsize=24, fontweight='bold')
axes[0].set_ylabel("Sub-wavelet Name", fontsize=24, fontweight='bold')
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 重新绘制Cd相关性热图，并添加格子
ax = sns.heatmap(correlations_cd, cmap=sns.color_palette("vlag", as_cmap=True), vmin=0, vmax=1, ax=axes[1],
            xticklabels=metric_names, yticklabels=subwavelet_names, cbar_kws={'label': 'Absolute Correlation'},
            linewidths=.5, linecolor='white')  # 添加白色格子
cbar = ax.collections[0].colorbar
cbar.set_label('Absolute Correlation', fontsize=20, fontweight='bold')
axes[1].set_title("Correlations with Cd", fontsize=24, fontweight='bold')
axes[1].set_xlabel("Metrics", fontsize=24, fontweight='bold')
axes[1].set_ylabel("Sub-wavelet Name", fontsize=24, fontweight='bold')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")



plt.tight_layout()
plt.savefig("wavelet_correl_heatmap.png", dpi=600)
plt.show()

