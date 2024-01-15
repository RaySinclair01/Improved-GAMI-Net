import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.signal import hilbert

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# 读取数据
file_path = "Hyperspectral_data.xlsx"
sheet_data = pd.read_excel(file_path, sheet_name=0)
sheet_data_2 = pd.read_excel(file_path, sheet_name=1)

# 定义波长范围
wavelengths = range(325, 1076)
wavelengths_sheet2 = range(325, 1075)

# 定义计算分组平均值差的绝对值的函数
def calculate_group_difference(sheet_data, wavelengths_range):
    group1 = sheet_data[sheet_data['group'].isin(['clean', 'low'])]
    group2 = sheet_data[sheet_data['group'].isin(['medium', 'high'])]
    differences = []
    for wl in wavelengths_range:
        avg1 = group1[wl].mean()
        avg2 = group2[wl].mean()
        differences.append(abs(avg2 - avg1))
    return pd.DataFrame({'Wavelength': wavelengths_range, 'Difference': differences}).sort_values('Difference', ascending=False)

# 计算全组的每个波长与叶绿素和镉的相关系数
correlations_with_cd = []
correlations_with_chl = []
for wl in wavelengths:
    corr_cd, _ = pearsonr(sheet_data[wl], sheet_data['Cd'])
    corr_chl, _ = pearsonr(sheet_data[wl], sheet_data['Chl'])
    correlations_with_cd.append(abs(corr_cd))
    correlations_with_chl.append(abs(corr_chl))

# 创建DataFrame并排序
sorted_correlations_with_chl = pd.DataFrame({'Wavelength': wavelengths, 'Abs_Correlation_with_Chl': correlations_with_chl}).sort_values('Abs_Correlation_with_Chl', ascending=False)
sorted_correlations_with_cd = pd.DataFrame({'Wavelength': wavelengths, 'Abs_Correlation_with_Cd': correlations_with_cd}).sort_values('Abs_Correlation_with_Cd', ascending=False)

# 准备热力图所需的数据
heatmap_data_chl = sorted_correlations_with_chl[['Wavelength', 'Abs_Correlation_with_Chl']].set_index('Wavelength').T
heatmap_data_cd = sorted_correlations_with_cd[['Wavelength', 'Abs_Correlation_with_Cd']].set_index('Wavelength').T
group_difference_1 = calculate_group_difference(sheet_data, wavelengths)
group_difference_2 = calculate_group_difference(sheet_data_2, wavelengths_sheet2)
heatmap_data_diff_1 = group_difference_1[['Wavelength', 'Difference']].set_index('Wavelength').T
heatmap_data_diff_2 = group_difference_2[['Wavelength', 'Difference']].set_index('Wavelength').T
# 定义去除包络线的函数
def remove_envelope_hilbert(spectrum):
    analytic_signal = hilbert(spectrum)
    envelope = np.abs(analytic_signal)
    envelope_removed_spectrum = spectrum - envelope
    return envelope_removed_spectrum

# 对每个样本做完包络线去除工作后，把“clean”和“low”组合并为一组，“medium”和“high”组合并为第二组
envelope_removed_data = sheet_data.iloc[:, 4:].apply(remove_envelope_hilbert, axis=1)
sheet_data_with_envelope_removed = sheet_data.copy()
sheet_data_with_envelope_removed.iloc[:, 4:] = envelope_removed_data
group_difference_envelope_removed = calculate_group_difference(sheet_data_with_envelope_removed, wavelengths)
heatmap_data_diff_envelope_removed = group_difference_envelope_removed[['Wavelength', 'Difference']].set_index('Wavelength').T
# 定义取对数倒数的函数
def log_inverse(spectrum):
    return -np.log(spectrum)

# 对每个样本做对数倒数处理后，把“clean”和“low”组合并为一组，“medium”和“high”组合并为第二组
log_inverse_data = sheet_data.iloc[:, 4:].apply(log_inverse, axis=1)
sheet_data_with_log_inverse = sheet_data.copy()
sheet_data_with_log_inverse.iloc[:, 4:] = log_inverse_data
group_difference_log_inverse = calculate_group_difference(sheet_data_with_log_inverse, wavelengths)
heatmap_data_diff_log_inverse = group_difference_log_inverse[['Wavelength', 'Difference']].set_index('Wavelength').T

# 创建可视化图形
plt.figure(figsize=(12, 6))

plt.subplot(3, 2, 1)
sns.heatmap(heatmap_data_chl, cmap="YlGnBu", cbar_kws={'label': 'Absolute Correlation'})
plt.title('Absolute Correlation with Chlorophyll')

plt.subplot(3, 2, 2)
sns.heatmap(heatmap_data_cd, cmap="YlOrRd", cbar_kws={'label': 'Absolute Correlation'})
plt.title('Absolute Correlation with Cadmium')

plt.subplot(3, 2, 3)
sns.heatmap(heatmap_data_diff_1, cmap="YlGnBu", cbar_kws={'label': 'Absolute Difference'})
plt.title('Absolute Difference between Grouped Averages (Sheet 1)')

plt.subplot(3, 2, 4)
sns.heatmap(heatmap_data_diff_2, cmap="YlOrRd", cbar_kws={'label': 'Absolute Difference'})
plt.title('Absolute Difference between Grouped Averages (Sheet 2)')

plt.subplot(3, 2, 5)
sns.heatmap(heatmap_data_diff_envelope_removed, cmap="YlGnBu", cbar_kws={'label': 'Absolute Difference'})
plt.title('Absolute Difference Envelope Removed')

# 子图B：对数倒数后的相关性热图
plt.subplot(3, 2, 6)
sns.heatmap(heatmap_data_diff_log_inverse, cmap="YlOrRd", cbar_kws={'label': 'Absolute Difference'})
plt.title('Absolute Difference Logarithmic Inverse')

plt.tight_layout()
plt.savefig("p12_Correlation_Difference_Envelope_Removal_Log_Inverse.png", dpi=600)
plt.show()
