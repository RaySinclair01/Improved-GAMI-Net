import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
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

# 创建可视化图形
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.heatmap(heatmap_data_chl, cmap="YlGnBu", cbar_kws={'label': 'Absolute Correlation'})
plt.title('Absolute Correlation with Chlorophyll')

plt.subplot(2, 2, 2)
sns.heatmap(heatmap_data_cd, cmap="YlOrRd", cbar_kws={'label': 'Absolute Correlation'})
plt.title('Absolute Correlation with Cadmium')

plt.subplot(2, 2, 3)
sns.heatmap(heatmap_data_diff_1, cmap="YlGnBu", cbar_kws={'label': 'Absolute Difference'})
plt.title('Absolute Difference between Grouped Averages (Sheet 1)')

plt.subplot(2, 2, 4)
sns.heatmap(heatmap_data_diff_2, cmap="YlOrRd", cbar_kws={'label': 'Absolute Difference'})
plt.title('Absolute Difference between Grouped Averages (Sheet 2)')

plt.tight_layout()
plt.savefig("Correlation_coefficient_ordering+heat_map_of band_differences.png", dpi=600)
plt.show()
