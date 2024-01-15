import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12  # 设定字体大小，例如12号字体
# 读取数据
file_path = "屈原光谱数据3-fa1.xlsx"
data = pd.read_excel(file_path, sheet_name=0)

# 选择波长范围
wavelengths_range = range(325, 1076)
spectra_data = data.iloc[:, 4:]

# 定义小波变换函数
def wavelet_transform(data, wavelet_name, level):
    coeffs = pywt.wavedec(data, wavelet_name, level=level)
    return coeffs

# 定义计算指标的函数
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

# 定义排序相关性的函数
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

# 将相关性值与标签组合并排序
sorted_correlations_cd, sorted_labels_cd = sort_correlations(correlations_cd, subwavelet_names, metric_names)
sorted_correlations_chl, sorted_labels_chl = sort_correlations(correlations_chl, subwavelet_names, metric_names)
# 创建一个数据框来保存四个变量
final_df = pd.DataFrame({
    'Sorted Labels Cd': sorted_labels_cd,
    'Sorted Correlations Cd': sorted_correlations_cd,
    'Sorted Labels Chl': sorted_labels_chl,
    'Sorted Correlations Chl': sorted_correlations_chl
})

# 输出数据框
print(final_df)

# 如果需要，也可以将数据框保存为Excel或CSV文件
final_df.to_excel('final_wavelet_output.xlsx', index=False)