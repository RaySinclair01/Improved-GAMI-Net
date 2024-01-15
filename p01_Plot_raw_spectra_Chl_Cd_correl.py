import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def calculate_correlation(group):
    correlations_with_cd = []
    correlations_with_chl = []
    wavelengths = range(325, 1076)
    for wl in wavelengths:
        corr_cd, _ = pearsonr(group[wl], group['Cd'])
        corr_chl, _ = pearsonr(group[wl], group['Chl'])
        correlations_with_cd.append(abs(corr_cd))
        correlations_with_chl.append(abs(corr_chl))
    return pd.DataFrame({
        'Wavelength': wavelengths,
        'Correlation_with_Cd': correlations_with_cd,
        'Correlation_with_Chl': correlations_with_chl
    })

def plot_with_overall_correlations(correlations_by_group, overall_correlations):
    plt.figure(figsize=(20, 10))
    font = {'family': 'Times New Roman', 'weight': 'bold'}
    plt.rc('font', **font)
    plt.subplot(2, 5, 1)
    plt.plot(overall_correlations['Wavelength'], overall_correlations['Correlation_with_Cd'], label='Correlation with Cd')
    plt.title('Overall (Cd)')
    plt.xlabel('Wavelength')
    plt.ylabel('Correlation')
    plt.legend()
    plt.subplot(2, 5, 6)
    plt.plot(overall_correlations['Wavelength'], overall_correlations['Correlation_with_Chl'], label='Correlation with Chl', color='orange')
    plt.title('Overall (Chl)')
    plt.xlabel('Wavelength')
    plt.ylabel('Correlation')
    plt.legend()
    for i, (group_name, correlations) in enumerate(correlations_by_group.items()):
        plt.subplot(2, 5, i + 2)
        plt.plot(correlations['Wavelength'], correlations['Correlation_with_Cd'], label='Correlation with Cd')
        plt.title(f'{group_name} group (Cd)')
        plt.xlabel('Wavelength')
        plt.ylabel('Correlation')
        plt.legend()
    for i, (group_name, correlations) in enumerate(correlations_by_group.items()):
        plt.subplot(2, 5, i + 7)
        plt.plot(correlations['Wavelength'], correlations['Correlation_with_Chl'], label='Correlation with Chl', color='orange')
        plt.title(f'{group_name} group (Chl)')
        plt.xlabel('Wavelength')
        plt.ylabel('Correlation')
        plt.legend()
    plt.tight_layout()
    plt.savefig("Plot_raw_spectra_Chl_Cd_correl.png", dpi=600)
    plt.show()

file_path = "Hyperspectral_data.xlsx" # 请更改为 文件路径
sheet_data = pd.read_excel(file_path, sheet_name=0)
grouped_data = sheet_data.groupby('group')
correlations_by_group = {group_name: calculate_correlation(group_data) for group_name, group_data in grouped_data}
group_order = ['clean', 'low', 'medium', 'high']
ordered_correlations_by_group = {group: correlations_by_group[group] for group in group_order}
overall_correlations = calculate_correlation(sheet_data)
plot_with_overall_correlations(ordered_correlations_by_group, overall_correlations)
