
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the Excel file and read the first worksheet
file_path = 'e41_Hyperspectral_data.xlsx'
df = pd.read_excel(file_path, sheet_name=0)

# Extract columns from 325 to 1075 (these are the wavelengths in nm)
wavelength_columns = [i for i in range(325, 1076)]
df_wavelengths = df[wavelength_columns]

# Split the data into train and test sets with a 4:1 ratio
df_train, df_test = train_test_split(df_wavelengths, test_size=0.2, random_state=42)

# Perform PCA to reduce dimensions to 3
pca = PCA(n_components=3)
train_reduced = pca.fit_transform(df_train)
test_reduced = pca.transform(df_test)

# Create a 3D scatter plot with updated axis limits and projections on specified planes (z = -0.3, x = -3, y = 0.6)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the new limits for the axes
ax.set_xlim([-4, 4])
ax.set_ylim([-0.5, 1])
ax.set_zlim([-0.4, 0.3])
ax.set_xticks(np.arange(-4, 4, 1.5))
ax.set_yticks(np.arange(-0.5, 1, 0.5))
ax.set_zticks(np.arange(-0.4, 0.3, 0.172))
# Style axis ticks
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(14)
    tick.set_fontweight('bold')

for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(14)
    tick.set_fontweight('bold')

for tick in ax.get_zticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(14)
    tick.set_fontweight('bold')
# Plot training data
ax.scatter(train_reduced[:, 0], train_reduced[:, 1], train_reduced[:, 2], c='#1f77b4',
           marker='o', s=100, alpha=0.8, edgecolors='k', linewidths=1, label='Train Set')

# Plot projections for training data on the specified planes
for point in train_reduced:
    # Projection to z = -0.4
    ax.plot([point[0], point[0]], [point[1], point[1]], [-0.4, point[2]], c='#1f77b4', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], point[1], -0.4, c='#1f77b4', marker='o', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to x = -4
    ax.plot([-4, point[0]], [point[1], point[1]], [point[2], point[2]], c='#1f77b4', linestyle='--',alpha=0.5, linewidth=0.7)
    ax.scatter(-4, point[1], point[2], c='#1f77b4', marker='o', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to y = 1
    ax.plot([point[0], point[0]], [1, point[1]], [point[2], point[2]], c='#1f77b4', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], 1, point[2], c='#1f77b4', marker='o', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)

# Plot test data
ax.scatter(test_reduced[:, 0], test_reduced[:, 1], test_reduced[:, 2], c='#ff7f0e',
           marker='^', s=100, alpha=0.8, edgecolors='k', linewidths=1, label='Test Set')

# Plot projections for test data on the specified planes
for point in test_reduced:
    # Projection to z = -0.3
    ax.plot([point[0], point[0]], [point[1], point[1]], [-0.3, point[2]], c='#ff7f0e', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], point[1], -0.3, c='#ff7f0e', marker='^', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to x = -3
    ax.plot([-3, point[0]], [point[1], point[1]], [point[2], point[2]], c='#ff7f0e', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(-3, point[1], point[2], c='#ff7f0e', marker='^', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to y = 0.6
    ax.plot([point[0], point[0]], [0.6, point[1]], [point[2], point[2]], c='#ff7f0e', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], 0.6, point[2], c='#ff7f0e', marker='^', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)

# Add labels and title with more stylized fonts
font_dict = {'family': 'Times New Roman', 'color':  'black', 'weight': 'normal', 'size': 14 }
font_dict2 = {'family': 'Times New Roman', 'color':  'black', 'weight': 'bold', 'size': 18 }
ax.set_xlabel('PC 1', fontdict=font_dict2)
ax.set_ylabel('PC 2', fontdict=font_dict2)
ax.set_zlabel('PC 3', fontdict=font_dict2)
#ax.set_title('Stylized 3D Visualization with Updated Axis Limits and Projections', fontdict=font_dict)
ax.legend(loc=(0.7, 0.8), frameon=False)

ax.xaxis.line.set_color('black')
ax.yaxis.line.set_color('black')
ax.zaxis.line.set_color('black')


ax.grid(color='lightgrey', linewidth=0.7)
plt.savefig('3d_scatter_projections.png', dpi=600)#, bbox_inches='tight')
# Show plot
plt.show()
