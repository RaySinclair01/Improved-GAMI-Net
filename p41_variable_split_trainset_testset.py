
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the new CSV file
file_path = 'GAMI-NET_Chl.csv'
df = pd.read_csv(file_path)

# Normalize the columns
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Split the data into train and test sets
df_train, df_test = train_test_split(df_normalized, test_size=0.2, random_state=42)

# Perform PCA to reduce dimensions to 3
pca = PCA(n_components=3)
train_reduced = pca.fit_transform(df_train)
test_reduced = pca.transform(df_test)

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Update the axis limits based on statistics
min_values = np.min(train_reduced, axis=0)
max_values = np.max(train_reduced, axis=0)
realm1=max_values[0]-min_values[0]
realm2=max_values[1]-min_values[1]
realm3=max_values[2]-min_values[2]
ax.set_xlim([min_values[0]-0.6*realm1, max_values[0]+0.6*realm1])
ax.set_ylim([min_values[1]-0.6*realm2, max_values[1]+0.6*realm2])
ax.set_zlim([min_values[2]-0.6*realm3, max_values[2]+0.6*realm3])
ax.set_xticks(np.arange(min_values[0]-0.6*realm1, max_values[0]+0.6*realm1, 0.6*realm1))
ax.set_yticks(np.arange(min_values[1]-0.6*realm2, max_values[1]+0.6*realm2, 0.6*realm2))
ax.set_zticks(np.arange(min_values[2]-0.6*realm3, max_values[2]+0.6*realm3, 0.54*realm3))
# Style axis ticks
font_dict = {'family': 'Times New Roman', 'color': 'black', 'weight': 'bold', 'size': 14}
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(14)
    tick.set_fontweight('bold')
    tick.set_color('black')

for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(14)
    tick.set_fontweight('bold')
    tick.set_color('black')

for tick in ax.get_zticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(14)
    tick.set_fontweight('bold')
    tick.set_color('black')

# Plot training and test data
ax.scatter(train_reduced[:, 0], train_reduced[:, 1], train_reduced[:, 2], c='#1f77b4',
           marker='o', s=100, alpha=0.8, edgecolors='k', linewidths=1, label='Train Set')

# Plot projections for training data on the specified planes
for point in train_reduced:
    # Projection to z = min_values[2]-0.6*realm3
    ax.plot([point[0], point[0]], [point[1], point[1]], [min_values[2]-0.6*realm3, point[2]], c='#1f77b4', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], point[1], min_values[2]-0.6*realm3, c='#1f77b4', marker='o', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to x = min_values[0]-0.6*realm1
    ax.plot([min_values[0]-0.6*realm1, point[0]], [point[1], point[1]], [point[2], point[2]], c='#1f77b4', linestyle='--',alpha=0.5, linewidth=0.7)
    ax.scatter(min_values[0]-0.6*realm1, point[1], point[2], c='#1f77b4', marker='o', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to y = max_values[1]+0.6*realm2
    ax.plot([point[0], point[0]], [max_values[1]+0.6*realm2, point[1]], [point[2], point[2]], c='#1f77b4', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], max_values[1]+0.6*realm2, point[2], c='#1f77b4', marker='o', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)

ax.scatter(test_reduced[:, 0], test_reduced[:, 1], test_reduced[:, 2], c='#ff7f0e',
           marker='^', s=100, alpha=0.8, edgecolors='k', linewidths=1, label='Test Set')

# Plot projections for test data on the specified planes
for point in test_reduced:
    # Projection to z = min_values[2]-0.6*realm3
    ax.plot([point[0], point[0]], [point[1], point[1]], [min_values[2]-0.6*realm3, point[2]], c='#ff7f0e', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], point[1], min_values[2]-0.6*realm3, c='#ff7f0e', marker='^', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to x = min_values[0]-0.6*realm1
    ax.plot([min_values[0]-0.6*realm1, point[0]], [point[1], point[1]], [point[2], point[2]], c='#ff7f0e', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(min_values[0]-0.6*realm1, point[1], point[2], c='#ff7f0e', marker='^', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)
    # Projection to y = max_values[1]+0.6*realm2
    ax.plot([point[0], point[0]], [max_values[1]+0.6*realm2, point[1]], [point[2], point[2]], c='#ff7f0e', linestyle='--', alpha=0.5,linewidth=0.7)
    ax.scatter(point[0], max_values[1]+0.6*realm2, point[2], c='#ff7f0e', marker='^', s=50, alpha=0.4, edgecolors='k', linewidths=0.5)

# Add labels and title with more stylized fonts
font_dict = {'family': 'Times New Roman',  'weight': 'bold', 'size': 17 }
font_dict2 = {'family': 'Times New Roman', 'color':  'black', 'weight': 'bold', 'size': 18 }
ax.set_xlabel('PC 1', fontdict=font_dict2, labelpad=10)
ax.set_ylabel('PC 2', fontdict=font_dict2, labelpad=10)
ax.set_zlabel('PC 3', fontdict=font_dict2, labelpad=10)

ax.legend(loc=(0.65, 0.82), frameon=False, prop=font_dict)

ax.xaxis.line.set_color('black')
ax.yaxis.line.set_color('black')
ax.zaxis.line.set_color('black')

ax.grid(color='lightgrey', linewidth=0.7)
# Save the figure
plt.savefig('p41_variable_split_trainset_testset.png', dpi=600)
# Show plot
plt.show()
# Create a pandas excel writer object to save the data
with pd.ExcelWriter('df_train_test_set.xlsx',engine='xlsxwriter') as writer:
    df_train.to_excel(writer, sheet_name='Train_Set', index=True)
    df_test.to_excel(writer, sheet_name='Test_Set', index=True)