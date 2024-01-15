import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 加载第一个工作表的数据
file_path = 'Hyperspectral_data.xlsx'
sheet1_data = pd.read_excel(file_path, sheet_name=0)

# 选择需要的列（波段波长对应的光谱反射率）
spectra_data = {
    'clean': sheet1_data[sheet1_data['group'] == 'clean'].iloc[:, 4:],
    'low': sheet1_data[sheet1_data['group'] == 'low'].iloc[:, 4:],
    'medium': sheet1_data[sheet1_data['group'] == 'medium'].iloc[:, 4:],
    'high': sheet1_data[sheet1_data['group'] == 'high'].iloc[:, 4:]
}

# 定义颜色
colors = {'clean': 'turquoise', 'low': 'yellow', 'medium': 'red', 'high': 'darkred'}

# 整合数据和组标签
all_data = pd.concat([spectra_data['clean'], spectra_data['low'], spectra_data['medium'], spectra_data['high']], axis=0)
group_labels = ['clean'] * len(spectra_data['clean']) + ['low'] * len(spectra_data['low']) + ['medium'] * len(spectra_data['medium']) + ['high'] * len(spectra_data['high'])

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(all_data)
tsne_df = pd.DataFrame(tsne_results, columns=['Component 1', 'Component 2'])
tsne_df['Group'] = group_labels

# 执行LDA降维
lda = LinearDiscriminantAnalysis()
lda_results = lda.fit_transform(all_data, group_labels)
lda_df = pd.DataFrame(lda_results, columns=['Component 1', 'Component 2', 'Component 3'])
lda_df['Group'] = group_labels

# 定义MSC和SNV函数
def msc_correction(spectrum):
    x = np.arange(len(spectrum))
    slope, intercept = np.polyfit(x, spectrum, 1)
    corrected_spectrum = spectrum - (slope * x + intercept)
    return corrected_spectrum

def snv_transformation(spectrum):
    mean = spectrum.mean()
    std_dev = spectrum.std()
    transformed_spectrum = (spectrum - mean) / std_dev
    return transformed_spectrum

# 对每个组的平均光谱进行MSC和SNV处理
msc_corrected_spectra = {group: msc_correction(data.mean()) for group, data in spectra_data.items()}
snv_transformed_spectra = {group: snv_transformation(data.mean()) for group, data in spectra_data.items()}

# 创建画布和子图
fig, axes = plt.subplots(2, 2, figsize=[18, 12])

# 绘制t-SNE子图
for group, color in colors.items():
    subset = tsne_df[tsne_df['Group'] == group]
    axes[0, 0].scatter(subset['Component 1'], subset['Component 2'], c=color, label=group, alpha=0.7)
axes[0, 0].set_title('t-SNE Visualization', fontsize=14)
axes[0, 0].set_xlabel('Component 1', fontsize=12)
axes[0, 0].set_ylabel('Component 2', fontsize=12)
axes[0, 0].legend(fontsize=10)

# 绘制LDA子图
for group, color in colors.items():
    subset = lda_df[lda_df['Group'] == group]
    axes[0, 1].scatter(subset['Component 1'], subset['Component 2'], c=color, label=group, alpha=0.7)
axes[0, 1].set_title('LDA Visualization', fontsize=14)
axes[0, 1].set_xlabel('Component 1', fontsize=12)
axes[0, 1].set_ylabel('Component 2', fontsize=12)
axes[0, 1].legend(fontsize=10)

# 绘制MSC子图
for group, color in colors.items():
    axes[1, 0].plot(msc_corrected_spectra[group], color=color, label=f'{group} group')
axes[1, 0].set_title('MSC Corrected Spectra', fontsize=14)
axes[1, 0].set_xlabel('Wavelength', fontsize=12)
axes[1, 0].set_ylabel('Reflectance', fontsize=12)
axes[1, 0].legend(fontsize=10)

# 绘制SNV子图
for group, color in colors.items():
    axes[1, 1].plot(snv_transformed_spectra[group], color=color, label=f'{group} group')
axes[1, 1].set_title('SNV Transformed Spectra', fontsize=14)
axes[1, 1].set_xlabel('Wavelength', fontsize=12)
axes[1, 1].set_ylabel('Reflectance', fontsize=12)
axes[1, 1].legend(fontsize=10)

# 调整子图间距
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()
