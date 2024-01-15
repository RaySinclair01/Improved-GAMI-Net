import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# 找出全局最大值和局部极值
from scipy.signal import find_peaks
from matplotlib import cm

deepest_blue = cm.coolwarm(0)  # 表示颜色条的最低端 蓝色
deepest_red = cm.coolwarm(0.9)  # 表示颜色条的最高端 红色

# Load the Excel file
file_path = 'e41_屈原光谱数据3-fa1.xlsx'

# Load the four worksheets into dataframes
df_raw_spectra = pd.read_excel(file_path, sheet_name='原始光谱')
df_diff_spectra = pd.read_excel(file_path, sheet_name='微分光谱')
df_envelope_removed = pd.read_excel(file_path, sheet_name='包络线去除')
df_inverse_log = pd.read_excel(file_path, sheet_name='取倒数对数')

# Dataframes to loop through
dfs = [df_raw_spectra, df_diff_spectra, df_envelope_removed, df_inverse_log]
# English names for the sheet names
sheet_names_english = ["Raw Spectrum", "Differential Spectrum", "Envelope Removal", "Inverse Log"]
# Chinese names for the sheet names
sheet_names = ['原始光谱', '微分光谱', '包络线去除', '取倒数对数']

# Colors for different groups
color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create a single legend for the top row
legend_labels = ['Clean', 'Low', 'Medium', 'High']
legend_colors = [color_sequence[i] for i in range(4)]
legend_patches = [plt.Rectangle((0,0), 1, 1, fc=legend_colors[i]) for i in range(4)]


# Initialize the new plot with further adjusted settings
fig, axs = plt.subplots(2, 4, figsize=(22, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.4)  # Increase vertical and horizontal space between subplots

# Updated font settings
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}

# Loop through each dataframe to create subplots for the first row
for idx, (df, sheet_name, english_name) in enumerate(zip(dfs, sheet_names, sheet_names_english)):
    plt.rc('font', **font)

    # Extract the spectral columns
    spectral_columns = [col for col in df.columns if isinstance(col, int)]

    # Group by 'group' and calculate the mean for each spectral column
    df_grouped_mean = df.groupby('group')[spectral_columns].mean().reset_index()
    df_grouped_mean = df_grouped_mean.set_index('group').loc[['clean', 'low', 'medium', 'high']].reset_index()

    # Calculate the mean difference spectra w.r.t 'clean'
    mean_diff = abs(df_grouped_mean[df_grouped_mean['group'] != 'clean'].set_index('group') - df_grouped_mean[
        df_grouped_mean['group'] == 'clean'].set_index('group').values)
    mean_diff_values = mean_diff.mean(axis=0)

    # Normalize the mean_diff_values
    mean_diff_values = (mean_diff_values - mean_diff_values.min()) / (mean_diff_values.max() - mean_diff_values.min())

    # Calculate min and max reflectance of the 'clean' group to position the color fill area
    t_clean = df_grouped_mean[df_grouped_mean['group'] == 'clean'][spectral_columns].values.flatten()
    min_t = min(t_clean)
    max_t = max(t_clean)
    t = max_t - min_t
    lower_bound = min_t + 0.2 * abs(t)
    upper_bound = min_t + 0.2 * abs(t) + 0.1 * abs(t)

    # Plot the color fill area in the middle, put it below the lines
    im = axs[0, idx].imshow([mean_diff_values],
                            extent=[min(spectral_columns), max(spectral_columns), lower_bound, upper_bound],
                            cmap='coolwarm', aspect='auto', zorder=1)

    # Plot the mean spectra for each group
    for i, group in enumerate(['clean', 'low', 'medium', 'high']):
        axs[0, idx].plot(spectral_columns,
                         df_grouped_mean[df_grouped_mean['group'] == group][spectral_columns].values.flatten(),
                         color=color_sequence[i], linewidth=2, zorder=0)
    # 新增代码：加粗所有子图的外框
    for ax in axs.flatten():
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

    # 新增代码：第一行第二个子图的纵轴设置保留两位小数
    axs[0, 1].yaxis.set_major_formatter('{:.2f}'.format)

    # 新增代码：加粗并放大所有子图的横纵轴刻度的数字
    for ax in axs.flatten():
        for ticklabel in (ax.get_xticklabels() + ax.get_yticklabels()):
            ticklabel.set_fontname('Times New Roman')
            ticklabel.set_fontsize(16)  # 选择合适的大小
            ticklabel.set_fontweight('bold')
    # Labels and title
    axs[0, idx].set_xlabel('Wavelength (nm)', fontdict=font)
    axs[0, idx].set_ylabel('Reflectance', fontdict=font)
    axs[0, idx].yaxis.set_major_formatter('{:.1f}'.format)  # One decimal place for y-axis labels
    axs[0, idx].set_title(sheet_names_english[idx], fontdict=font)

    # Add a small colorbar for each subplot in the first row
    divider = make_axes_locatable(axs[0, idx])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)  # Increase colorbar label size

    # Second row: correlation with Chl
    corr_with_chl = df[spectral_columns].apply(lambda x: np.corrcoef(x, df['Chl'])[0, 1], axis=0)

    # Plot the correlation line in purple
    axs[1, idx].plot(spectral_columns, corr_with_chl, color=deepest_blue, linewidth=2, zorder=1)

    # 计算全局最大值
    global_max_idx = np.argmax(np.abs(corr_with_chl))
    # 找出局部极值
    peaks, _ = find_peaks(np.abs(corr_with_chl))
    # 保证peaks中的所有值都在corr_with_chl的有效索引范围内
    peaks = peaks[peaks < len(corr_with_chl)]
    # 按照相关系数的绝对值大小对局部极大值进行排序，然后取前10%
    sorted_peaks = peaks[np.argsort(-np.abs(corr_with_chl.iloc[peaks]))]  # 使用 .iloc[] 进行位置索引
    top_10_percent_peaks = sorted_peaks[:int(len(sorted_peaks) * 0.1)]
    # 组合全局最大值和前10%的局部极大值
    important_indices = np.unique(np.concatenate(([global_max_idx], top_10_percent_peaks)))
    # 在重要索引处添加淡绿色垂直阴影
    for i in important_indices:
        # 假设 axs[1, idx] 是正确的方式来引用子图。这将取决于你的子图是如何组织的。
        axs[1, idx].axvspan(spectral_columns[i] - 2, spectral_columns[i] + 2, facecolor=deepest_red, alpha=0.8)

    axs[1, idx].set_xlabel('Wavelength (nm)', fontdict=font)
    axs[1, idx].set_ylabel('Correlation Coeff.', fontdict=font)
    axs[1, idx].set_title(sheet_names_english[idx], fontdict=font)

# Create a single legend for the top row, without a frame and closer to the plots
legend = axs[0, 0].legend(legend_patches, legend_labels, loc='upper left', ncol=1, fontsize='small', frameon=False)
plt.savefig('e41_band_compare.png', dpi=600, bbox_inches='tight')
# Show the plot
plt.show()
