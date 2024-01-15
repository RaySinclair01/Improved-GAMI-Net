import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14
# Load the updated data file
file_path_updated = '光谱特征参数+Cd和Chl加组名34.xlsx'  # Removed Cd
data_updated = pd.read_excel(file_path_updated)

# Initialize lists to store the data to be plotted
filtered_columns = []
correlations_chl_filtered = []

# Calculate correlations for Chl, ignoring columns with NaN or inf
for col in data_updated.columns[:21]:
    if not (data_updated[col].isnull().any() or data_updated['Chl'].isnull().any()):
        corr_chl = np.corrcoef(data_updated[col], data_updated['Chl'])[0, 1]
        if np.isfinite(corr_chl):
            filtered_columns.append(col)
            correlations_chl_filtered.append(corr_chl)

# Create a horizontal bar plot representing correlations with Chl
plt.figure(figsize=(10, 10))

bars = plt.barh(filtered_columns, correlations_chl_filtered, color='dodgerblue', alpha=0.7)

# Add data labels
for bar in bars:
    width = bar.get_width()
    label_x_pos = width if width >= 0 else width - 0.03
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

# Add grid, title, and labels
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.title('Correlations Between Spectral Features and Chlorophyll Content', fontsize=16)
plt.ylabel('Spectral Features', fontsize=14)
plt.xlabel('Correlation Coefficient', fontsize=14)

# Fine-tune the aesthetics
# Make all the spines visible and set their color to gray
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('gray')

plt.tight_layout()

# Save the figure with high resolution
plt.savefig("bar_correl_with_Chl_horizontal_academic_filtered_with_border.png", dpi=600)

plt.show()