import pandas as pd
import numpy as np
from scipy.signal import hilbert

# 定义去除包络线的函数
def remove_envelope_hilbert(spectrum):
    analytic_signal = hilbert(spectrum)
    envelope = np.abs(analytic_signal)
    envelope_removed_spectrum = spectrum - envelope
    return envelope_removed_spectrum

# 读取工作表
file_path = 'Hyperspectral_data.xlsx' # 请替换为你的文件路径
data = pd.read_excel(file_path, sheet_name=0)

# 选择光谱波段的列
spectrum_columns = data.columns[4:]

# 初始化包络线去除和取倒数对数的数据框
envelope_removed_data = data.copy()
reciprocal_log_data = data.copy()

# 对每一行样本进行包络线去除和取倒数对数的处理
for index, row in data.iterrows():
    spectrum_values = row[spectrum_columns].values.astype(float) # 转换为浮点数
    envelope_removed_spectrum = remove_envelope_hilbert(spectrum_values)
    envelope_removed_data.loc[index, spectrum_columns] = envelope_removed_spectrum
    reciprocal_log_spectrum = np.log(1 / spectrum_values)
    reciprocal_log_data.loc[index, spectrum_columns] = reciprocal_log_spectrum

# 分组并计算A、B组的包络线去除和取倒数对数的曲线
group_A = envelope_removed_data[(envelope_removed_data['group'] == 'clean') | (envelope_removed_data['group'] == 'low')]
group_B = envelope_removed_data[(envelope_removed_data['group'] == 'medium') | (envelope_removed_data['group'] == 'high')]
envelope_removed_A_mean = group_A[spectrum_columns].mean()
envelope_removed_B_mean = group_B[spectrum_columns].mean()
envelope_removed_diff = np.abs(envelope_removed_A_mean - envelope_removed_B_mean)

group_A_log = reciprocal_log_data[(reciprocal_log_data['group'] == 'clean') | (reciprocal_log_data['group'] == 'low')]
group_B_log = reciprocal_log_data[(reciprocal_log_data['group'] == 'medium') | (reciprocal_log_data['group'] == 'high')]
reciprocal_log_A_mean = group_A_log[spectrum_columns].mean()
reciprocal_log_B_mean = group_B_log[spectrum_columns].mean()
reciprocal_log_diff = np.abs(reciprocal_log_A_mean - reciprocal_log_B_mean)

# 创建数据框C
data_frame_C = pd.DataFrame({
    'Wavelength': spectrum_columns,
    'Envelope_Removed_Diff': envelope_removed_diff.values,
    'Reciprocal_Log_Diff': reciprocal_log_diff.values
})

# 按第二列和第三列的数值进行降序排序
sorted_by_envelope_removed = data_frame_C.sort_values(by='Envelope_Removed_Diff', ascending=False)
sorted_by_reciprocal_log = data_frame_C.sort_values(by='Reciprocal_Log_Diff', ascending=False)

# 将各自排序后排在前三位置的波段名存储到新的数据框D
data_frame_D = pd.DataFrame({
    'Envelope_Removed': sorted_by_envelope_removed['Wavelength'].head(3).values,
    'Reciprocal_Log': sorted_by_reciprocal_log['Wavelength'].head(3).values
})
