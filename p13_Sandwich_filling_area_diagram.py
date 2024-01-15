import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12  # 设定字体大小，例如12号字体
# 读取数据
file_path = "Line_Data.csv"
df = pd.read_csv(file_path)

# 转换日期格式
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

# 计算差值
diff = df['AAPL'] - df['AMZN']

# 创建新的自定义 colormap
def custom_colormap():
    cdict = {
        'red':   [(0.0,  1.0, 1.0),
                  (0.5,  1.0, 1.0),
                  (1.0,  0.0, 0.0)],
        'green': [(0.0,  0.5, 0.5),
                  (0.5,  1.0, 1.0),
                  (1.0,  0.5, 0.5)],
        'blue':  [(0.0,  0.5, 0.5),
                  (0.5,  1.0, 1.0),
                  (1.0,  0.5, 0.5)]
    }
    return LinearSegmentedColormap('Custom_Colormap', segmentdata=cdict)

# 创建新的 colormap
colormap = custom_colormap()

# 创建新的 ScalarMappable 对象以使用新的 colormap
#norm = plt.Normalize(vmin=diff.min(), vmax=diff.max())
norm = plt.Normalize(vmin=-100, vmax=100)
sm = ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['AMZN'], label='AMZN', color='blue')
plt.plot(df['date'], df['AAPL'], label='AAPL', color='red')

# 使用新的 colormap 填充区域
for i in range(len(df['date']) - 1):
    plt.fill_between(df['date'][i:i+2], df['AMZN'][i:i+2], df['AAPL'][i:i+2],
                     color=colormap(norm(diff[i])), alpha=0.8)

# 添加标签、标题和图例
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Comparison Between AMZN and AAPL')
plt.legend()
plt.colorbar(sm, label='Difference')

# 设置日期格式
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# 显示图表
plt.show()

