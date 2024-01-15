# 导入所需库
import pandas as pd

# 读取Excel文件
file_path = "光谱特征参数+Cd和Chl加组名34.xlsx"  # 替换为 文件路径
data = pd.read_excel(file_path)

# 提取光谱特征和Chl和Cd列
spectral_features = data.iloc[:, :21]
Chl = data['Chl']
Cd = data['Cd']

# 计算Chl和Cd的相关系数的绝对值
correlation_with_Chl = spectral_features.corrwith(Chl).abs()
correlation_with_Cd = spectral_features.corrwith(Cd).abs()

# 创建数据框A
A = pd.DataFrame({
    'Spectral Feature': spectral_features.columns,
    'Correlation with Cd': correlation_with_Cd,
    'Correlation with Chl': correlation_with_Chl
})

# 按Cd和Chl的相关性进行排序，并提取与DataFrame B的前三个光谱特征
B_Cd = A.nlargest(3, 'Correlation with Cd')['Spectral Feature'].reset_index(drop=True)
B_Chl = A.nlargest(3, 'Correlation with Chl')['Spectral Feature'].reset_index(drop=True)

# 创建数据框B
B = pd.DataFrame({
    'Top 3 with Cd': B_Cd,
    'Top 3 with Chl': B_Chl
})

# 按Cd和Chl的相关性对数据框A进行排序
A_sorted_by_Cd = A.sort_values(by='Correlation with Cd', ascending=False).reset_index(drop=True)
A_sorted_by_Chl = A.sort_values(by='Correlation with Chl', ascending=False).reset_index(drop=True)

# 打印结果
print(A_sorted_by_Cd)
print(A_sorted_by_Chl)
print(B)
