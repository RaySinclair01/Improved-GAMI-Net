import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

# Load the data
data_filepath = '屈原光谱数据3-fa1.xlsx'
data = pd.read_excel(data_filepath, sheet_name=0)



# Define the groups
groups = ['clean', 'low', 'medium', 'high']

# Define wavelength regions to highlight based on the specified ranges
regions = {
    "Ultraviolet": (325, 400, 'violet'),
    "Violet-Blue": (400, 495, 'blue'),
    "Blue Edge": (450, 500, 'skyblue'),  # Approximate, will adjust based on data
    "Green": (495, 570, 'green'),
    "Green Peak": (500, 570, 'lime'),  # Approximate, will adjust based on data
    "Yellow Edge": (570, 590, 'yellow'),
    "Yellow": (570, 590, 'yellow'),
    "Orange": (590, 620, 'orange'),
    "Red": (620, 750, 'red'),
    "Red Valley": (670, 690, 'darkred'),  # Approximate, will adjust based on data
    "Red Edge": (700, 740, 'maroon'),
    "Near-Infrared (NIR)": (750, 1000, 'grey'),
    "Shortwave Infrared (SWIR)": (1000, 1075, 'black')
}

# Font properties
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('normal')
font.set_weight('bold')

# Create 1x4 subplots with adjustments based on feedback
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
# Create a list to store handles and labels for the legend
handles, labels = [], []
# Plot the spectral reflectance curves for each group and highlight specific regions
for i, group in enumerate(groups):
    # 获取该组的所有样本数据
    group_data = data[data['group'] == group]

    for j, row in group_data.iterrows():
        spectral_data = row.loc[325:1075]
        # 绘制数据
        axes[i].plot(spectral_data.index, spectral_data.values)

    # Highlight the specified regions
    for region, (start, end, color) in regions.items():
        axes[i].axvspan(start, end, color=color, alpha=0.2, label=region)

    # Set plot labels and title with specified font properties
    axes[i].set_xlabel('Wavelength (nm)', fontproperties=font, color='black',fontsize=20)
    axes[i].set_title(f'{group.capitalize()} Group', fontproperties=font, color='black',fontsize=24)

    # Adding grid
    axes[i].grid(True, which="both", ls="--", c='white')

    # Adding legend only to the last subplot
    if i == 3:
        axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', prop=font)

    # Adding a box around the subplots
    for _, spine in axes[i].spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

# Setting y-label only for the first subplot with specified font properties
axes[0].set_ylabel('Reflectance', fontproperties=font, color='black',fontsize=20)

# Adjust the layout so plots are not overlapping
plt.tight_layout()
# Save the figure
plt.savefig('grouped3.png', dpi=600)
# Display the plot
plt.show()
