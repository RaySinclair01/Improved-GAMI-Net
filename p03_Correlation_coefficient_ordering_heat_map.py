# 导入所需的库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
# 读取数据文件
file_path = "Hyperspectral_data.xlsx" # 请修改为 文件路径
sheet_data = pd.read_excel(file_path, sheet_name=0)

# 计算全组的每个波长与叶绿素和镉的相关系数
from scipy.stats import pearsonr

correlations_with_cd = []
correlations_with_chl = []
wavelengths = range(325, 1076)
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

# 创建可视化图形
plt.figure(figsize=(12, 5))

# 子图1：与叶绿素的相关系数排序的热力图
plt.subplot(1, 2, 1)
sns.heatmap(heatmap_data_chl, cmap="YlGnBu", cbar_kws={'label': 'Absolute Correlation'})
plt.title('Absolute Correlation with Chlorophyll')

# 子图2：与镉的相关系数排序的热力图
plt.subplot(1, 2, 2)
sns.heatmap(heatmap_data_cd, cmap="YlOrRd", cbar_kws={'label': 'Absolute Correlation'})
plt.title('Absolute Correlation with Cadmium')

plt.tight_layout()
plt.savefig("Correlation_coefficient_ordering_heat_map.png", dpi=600)
plt.show()
