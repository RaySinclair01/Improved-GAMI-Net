import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
excel_path = "Hyperspectral_data.xlsx"
df = pd.read_excel(excel_path,sheet_name=0)

# 计算Cd列的平均值和标准差
cd_mean = np.mean(df['Cd'])
cd_std = np.std(df['Cd'])
cd_min = np.min(df['Cd'])
cd_max = np.max(df['Cd'])
cd_skew = skew(df['Cd'])
cd_kurtosis = kurtosis(df['Cd'])
# 计算Chl列的平均值和标准差
chl_mean = np.mean(df['Chl'])
chl_std = np.std(df['Chl'])
chl_min = np.min(df['Chl'])
chl_max = np.max(df['Chl'])
chl_skew = skew(df['Chl'])
chl_kurtosis = kurtosis(df['Chl'])
# 使用Pandas创建一个数据框来存储这些统计数据
statistics_data = {
    'Metric': ['Mean', 'Standard Deviation', 'Min', 'Max','skew','kurtosis'],
    'Cd': [cd_mean, cd_std, cd_min, cd_max,cd_skew,cd_kurtosis],
    'Chl': [chl_mean, chl_std, chl_min, chl_max,chl_skew,chl_kurtosis]
}

# 创建数据框
statistics_df = pd.DataFrame(statistics_data)

# 显示数据框
statistics_df

