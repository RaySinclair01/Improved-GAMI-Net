
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def perform_pchip_interpolation(input_file, output_file, extrapolate=True):
    # Load the input Excel file
    df = pd.read_excel(input_file)

    # Sentinel-2 Band wavelengths in nm
    band_wavelengths = np.array([443, 490, 560, 665, 705, 740, 783, 865, 945, 1610, 2190])

    # Target wavelengths for interpolation (325-1075 nm)
    target_wavelengths = np.arange(325, 1076)

    # Initialize an empty dataframe to hold the interpolated values
    df_interpolated = pd.DataFrame(columns=[f'R{wl}' for wl in target_wavelengths])

    # Perform PCHIP interpolation for each row
    for index, row in df.iterrows():
        pchip = PchipInterpolator(band_wavelengths, row.values, extrapolate=extrapolate)  # Extrapolation enabled if true
        interpolated_values = pchip(target_wavelengths)
        # Replace negative values with the nearest positive value
        interpolated_values[interpolated_values < 0] = np.min(interpolated_values[interpolated_values >= 0])
        df_interpolated.loc[index] = interpolated_values

    # Save the interpolated dataframe to the output Excel file
    df_interpolated.to_excel(output_file, index=False)

# Example usage:
input_file_path = 'e52_extracted_xt_HSP_data_reflectance.xlsx'
output_file_path = 'interpolated_data_pchip_extrapolated.xlsx'
perform_pchip_interpolation(input_file_path, output_file_path, extrapolate=True)
