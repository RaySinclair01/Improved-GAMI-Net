import pandas as pd
import pywt
import matplotlib.pyplot as plt

# 加载第一个工作表的数据
file_path = '屈原光谱数据3-fa1.xlsx'
sheet1_data = pd.read_excel(file_path, sheet_name=0)

# 选择需要的列（波段波长对应的光谱反射率）
spectra_data = {
    'clean': sheet1_data[sheet1_data['group'] == 'clean'].iloc[:, 4:],
    'low': sheet1_data[sheet1_data['group'] == 'low'].iloc[:, 4:],
    'medium': sheet1_data[sheet1_data['group'] == 'medium'].iloc[:, 4:],
    'high': sheet1_data[sheet1_data['group'] == 'high'].iloc[:, 4:]
}

# 从clean组的平均值进行小波分解，并获取分解的层数
average_spectrum_clean = spectra_data['clean'].mean()
coeffs_clean = pywt.wavedec(average_spectrum_clean, 'db1')

# 将小波变换的分解结果绘制为一行的整图
fig, axes = plt.subplots(len(coeffs_clean), 4, figsize=[18, 10])

# 定义颜色
colors = {'clean': 'turquoise', 'low': 'yellow', 'medium': 'red', 'high': 'darkred'}

# 对每个组的平均值进行小波分解并绘制结果
for idx, (group, color) in enumerate(colors.items()):
    average_spectrum = spectra_data[group].mean()
    coeffs_group = pywt.wavedec(average_spectrum, 'db1')
    for i, coef in enumerate(coeffs_group):
        axes[i, idx].plot(coef, color=color)
        if idx == 0:
            axes[i, idx].set_title(f'Level {len(coeffs_group) - i - 1} Coefficients')
        axes[i, idx].grid(True)

# 设置组标签
axes[0, 0].set_ylabel('clean')
axes[0, 1].set_ylabel('low')
axes[0, 2].set_ylabel('medium')
axes[0, 3].set_ylabel('high')

plt.xlabel('Coefficient Index')
plt.tight_layout()
plt.savefig("wavelet.png", dpi=600)
plt.show()
