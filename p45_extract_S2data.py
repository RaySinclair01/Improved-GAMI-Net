import pandas as pd
import rasterio
from pyproj import Transformer
# 读取Excel文件
file_path = 'e02_Cd_xy.xls'
df_coordinates = pd.read_excel(file_path)
# 初始化一个空的DataFrame用于保存结果
df_result = pd.DataFrame()
# 创建坐标转换器
transformer = Transformer.from_crs(4326, 32649, always_xy=True)
# 设置转换参数
BOA_QUANTIFICATION_VALUE = 10000
U = 0.974015708241877
#S2data_path = 'F:/Program_Database/PaperDatabase/08 图片/遥感map图1314/S2B_MSIL2A_20190811T030549_N0213_R075_T49RFL_20190811T071404.SAFE/GRANULE/L2A_T49RFL_A012685_20190811T031421/IMG_DATA/R60m'
S2data_path = 'F:/Program_Database/PaperDatabase/08 图片/遥感map图1314/S2A_MSIL2A_20190816T030551_N0213_R075_T49RFL_20190816T071555.SAFE/GRANULE/L2A_T49RFL_A021665_20190816T031414/IMG_DATA/R60m'
# 读取所有需要的.jp2文件（替换成您本地的文件路径）
band_files = {
    'Band_1': f'{S2data_path}/T49RFL_20190816T030551_B01_60m.jp2',
    'Band_2': f'{S2data_path}/T49RFL_20190816T030551_B02_60m.jp2',
    'Band_3': f'{S2data_path}/T49RFL_20190816T030551_B03_60m.jp2',
    'Band_4': f'{S2data_path}/T49RFL_20190816T030551_B04_60m.jp2',
    'Band_5': f'{S2data_path}/T49RFL_20190816T030551_B05_60m.jp2',
    'Band_6': f'{S2data_path}/T49RFL_20190816T030551_B06_60m.jp2',
    'Band_7': f'{S2data_path}/T49RFL_20190816T030551_B07_60m.jp2',
    'Band_8': f'{S2data_path}/T49RFL_20190816T030551_B8A_60m.jp2',
    'Band_9': f'{S2data_path}/T49RFL_20190816T030551_B09_60m.jp2',
    'Band_11': f'{S2data_path}/T49RFL_20190816T030551_B11_60m.jp2',
    'Band_12': f'{S2data_path}/T49RFL_20190816T030551_B12_60m.jp2'
}

# 遍历所有经纬度点
for idx, row in df_coordinates.iterrows():
    lon, lat = row['经度'], row['纬度']

    # 转换坐标
    lon, lat = transformer.transform(lon, lat)

    row_data = {'ID': row['ID']}

    # 遍历所有波段
    for band, file_path in band_files.items():
        with rasterio.open(file_path) as src:
            # 获取转换矩阵
            raster_transform = src.transform

            # 将经纬度坐标转换为像素坐标
            x, y = ~raster_transform * (lon, lat)
            x, y = int(x), int(y)

            # 提取该点在该波段的值（DN）
            DN = src.read(1)[y, x]

            # 转换为反射率
            reflectance = (DN / BOA_QUANTIFICATION_VALUE) * U

            row_data[band] = reflectance

    # 将结果添加到DataFrame中
    df_result = pd.concat([df_result, pd.DataFrame([row_data])], ignore_index=True)

# 保存结果到Excel文件
df_result.to_excel("extracted_xt_HSP_data_reflectance.xlsx", index=False)