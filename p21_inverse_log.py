import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
# 加载数据
file_path = '屈原光谱数据3-fa1.xlsx'
sheet_data = pd.read_excel(file_path, sheet_name=0)
spectra_data=sheet_data.iloc[:, 4:]


# 定义取倒数对数的函数
def reciprocal_log(spectrum):
    return np.log(1 / spectrum.astype(float))

reciprocal_log_spectra_data_hilbert=spectra_data.apply(reciprocal_log, axis=1)
# 获取列名（波段名称）
spectra_columns = spectra_data.columns

# 创建一个新的空 DataFrame，索引为波段名称
final_dataframe = pd.DataFrame(index=spectra_columns)

# 遍历去除包络线后的光谱数据的每一行（即每个样本）
for sample_index, reciprocal_log_spectrum in reciprocal_log_spectra_data_hilbert.iterrows():
    final_dataframe[sample_index] = reciprocal_log_spectrum

# 最终，final_dataframe 包括所有样本的去除包络线后的光谱数据，每一列对应一个样本，列名为样本索引
# 转置 DataFrame
final_dataframe_transposed = final_dataframe.transpose()
final_dataframe_transposed.to_excel('屈原光谱数据3-fa1_取倒数对数.xlsx', index=True)