import pandas as pd

# Load the data from the first worksheet
file_path = "Hyperspectral_data.xlsx"

# Process the data from the first worksheet
def process_data(sheet, col_end):
    # Combine 'clean' and 'low' into group A, 'medium' and 'high' into group B
    group_A = sheet[(sheet['group'] == 'clean') | (sheet['group'] == 'low')]
    group_B = sheet[(sheet['group'] == 'medium') | (sheet['group'] == 'high')]

    # Calculate the average reflectance for each group
    wavelengths = list(range(325, col_end + 1))
    group_A_avg = group_A[wavelengths].mean()
    group_B_avg = group_B[wavelengths].mean()

    # Create a new dataframe with the average reflectance for each group
    avg_df = pd.DataFrame({
        'Wavelength': wavelengths,
        'Group_A_Reflectance_Average': group_A_avg,
        'Group_B_Reflectance_Average': group_B_avg
    })

    # Calculate the absolute difference between the average reflectance of group B and group A
    avg_df['A-B'] = abs(avg_df['Group_B_Reflectance_Average'] - avg_df['Group_A_Reflectance_Average'])

    # Sort the dataframe by the 'A-B' column in descending order
    avg_df_sorted = avg_df.sort_values(by='A-B', ascending=False)

    # Get the top 3 wavelengths
    top_3_wavelengths = avg_df_sorted['Wavelength'].head(3).values

    return top_3_wavelengths

# Load the data from the worksheets
sheet1 = pd.read_excel(file_path, sheet_name=0)
sheet2 = pd.read_excel(file_path, sheet_name=1)

# Get the top 3 wavelengths for the first worksheet
top_3_wavelengths_sheet1 = process_data(sheet1, 1075)
print("Top 3 wavelengths for the first worksheet:", top_3_wavelengths_sheet1)

# Get the top 3 wavelengths for the second worksheet
top_3_wavelengths_sheet2 = process_data(sheet2, 1074)
print("Top 3 wavelengths for the second worksheet:", top_3_wavelengths_sheet2)
out_df = pd.DataFrame({
    'Org_band': top_3_wavelengths_sheet1,
    'fo_band': top_3_wavelengths_sheet2
})