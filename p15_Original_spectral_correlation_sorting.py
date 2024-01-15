import pandas as pd
import numpy as np

# Load the data
data = pd.read_excel("Hyperspectral_data.xlsx", sheet_name=0)

# Create empty lists to store the wavelengths and correlations
wavelengths = []
correlations_with_cd = []
correlations_with_chl = []

# Calculate the correlations for each wavelength
for col in data.columns[4:]:
    wavelengths.append(int(col))
    correlations_with_cd.append(data['Cd'].corr(data[col]))
    correlations_with_chl.append(data['Chl'].corr(data[col]))

# Create a dataframe with the calculated correlations
correlation_df = pd.DataFrame({
    'Wavelength': wavelengths,
    'Correlation_with_Cd': correlations_with_cd,
    'Correlation_with_Chl': correlations_with_chl
})

# Sort the dataframe by Correlation_with_Cd in descending order and get the top 3 wavelengths
correlation_df_sorted_cd = correlation_df.sort_values(by='Correlation_with_Cd', ascending=False)
top3_wavelengths_cd = correlation_df_sorted_cd['Wavelength'].head(3).tolist()

# Sort the dataframe by Correlation_with_Chl in descending order and get the top 3 wavelengths
correlation_df_sorted_chl = correlation_df.sort_values(by='Correlation_with_Chl', ascending=False)
top3_wavelengths_chl = correlation_df_sorted_chl['Wavelength'].head(3).tolist()

# Combine the two lists of top 3 wavelengths
top3_wavelengths = top3_wavelengths_cd + top3_wavelengths_chl

# Create a dataframe with the combined top 3 wavelengths
top3_df = pd.DataFrame({
    'Top3_Wavelengths_Cd': top3_wavelengths_cd,
    'Top3_Wavelengths_Chl': top3_wavelengths_chl
})

top3_df
