
# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 定义用于多项式回归的函数
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

# 读取Excel文件
file_path = 'e41_Hyperspectral_data.xlsx'
df = pd.read_excel(file_path, sheet_name='原始光谱')

# 转换列名为字符串
df.columns = df.columns.astype(str)

# 计算指数值 Sum_Dr1A
indices_df = pd.DataFrame()
indices_df['Sum_Dr1A'] = df.loc[:, '625':'795'].diff(axis=1).sum(axis=1)

# 准备数据
X = indices_df['Sum_Dr1A'].values.reshape(-1, 1)
y = df['Chl'].values

# 尝试不同的多项式阶数，选择一个最优的
best_degree = 0
best_r2 = -np.inf
best_model = None

for degree in range(1, 6):
    model = PolynomialRegression(degree)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree
        best_model = model

# 使用最优模型进行预测
Chl_pre = best_model.predict(X)

# 将预测值加入到数据框中
df['Chl_pre'] = Chl_pre

# 计算性能指标
R2 = r2_score(df['Chl'], df['Chl_pre'])
RMSE = np.sqrt(mean_squared_error(df['Chl'], df['Chl_pre']))
RPD = df['Chl'].std() / RMSE

R2, RMSE, RPD
