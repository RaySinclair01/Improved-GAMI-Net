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

# 首先，将三维矩阵 three_dim_matrix 调整为一个二维形状，其中每一行是一个样本，每一列是子波和度量的唯一组合。
reshaped_matrix = three_dim_matrix.reshape(num_samples, num_subwaves * num_metrics)

# 生成列名，通过创建所有子波名和度量名的组合。
column_names = [f"{subwave}-{metric}" for subwave in subwavelet_names for metric in metric_names]

# 使用调整后的矩阵和生成的列名创建DataFrame。
final_df = pd.DataFrame(reshaped_matrix, columns=column_names)

# 显示结果DataFrame的一部分以验证它是否符合我们的预期。
print(final_df.head())

# 将DataFrame保存为Excel文件。
final_df.to_excel('three_dim_matrix_to_excel.xlsx', index=True)
