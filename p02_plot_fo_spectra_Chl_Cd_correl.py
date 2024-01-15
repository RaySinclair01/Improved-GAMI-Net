import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 读取数据
file_path = "Hyperspectral_data.xlsx"
sheet_data = pd.read_excel(file_path, sheet_name=0)

# 计算相关性的函数
def calculate_correlation(group):
    correlations_with_cd = []
    correlations_with_chl = []
    wavelengths = range(325, 1076)
    for wl in wavelengths:
        corr_cd, _ = pearsonr(group[wl], group['Cd'])
        corr_chl, _ = pearsonr(group[wl], group['Chl'])
        correlations_with_cd.append(abs(corr_cd))  # 提取相关系数值后再应用abs()函数
        correlations_with_chl.append(abs(corr_chl))  # 提取相关系数值后再应用
    return pd.DataFrame({
        'Wavelength': wavelengths,
        'Correlation_with_Cd': correlations_with_cd,
        'Correlation_with_Chl': correlations_with_chl
    })

# 按group分组，并计算每组的相关性
grouped_data = sheet_data.groupby('group')
correlations_by_group = {group_name: calculate_correlation(group_data) for group_name, group_data in grouped_data}

# 创建一个更大的图表用于绘制所有6个子图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 定义绘制每个子图的函数
def plot_subplot(ax, correlations, comparison_with, group_name, attribute, title):
    ax.plot(correlations[comparison_with]['Wavelength'], correlations[comparison_with][attribute], label=f'{comparison_with} group', color='blue', linewidth=0.5)
    ax.plot(correlations[group_name]['Wavelength'], correlations[group_name][attribute], label=f'{group_name} group', color='red', linewidth=0.5)
    ax.fill_between(correlations[comparison_with]['Wavelength'], correlations[comparison_with][attribute], correlations[group_name][attribute], color='gray', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel(f'Correlation with {attribute.split("_")[-1]}')
    ax.legend()

# 绘制与Cd的相关性对比
for idx, group_name in enumerate(['low', 'medium', 'high']):
    plot_subplot(axes[0, idx], correlations_by_group, 'clean', group_name, 'Correlation_with_Cd', f'Comparison with clean group for {group_name} group')

# 绘制与Chl的相关性对比
for idx, group_name in enumerate(['low', 'medium', 'high']):
    plot_subplot(axes[1, idx], correlations_by_group, 'clean', group_name, 'Correlation_with_Chl', f'Comparison with clean group for {group_name} group')

plt.tight_layout()
plt.savefig("Plot_raw_spectra_Chl_Cd_correl.png", dpi=600)
plt.show()
