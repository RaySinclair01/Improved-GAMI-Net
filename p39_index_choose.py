
import pandas as pd
import numpy as np

# Load the Excel file
file_path = 'e41_屈原光谱数据3-fa1.xlsx'
df_first_sheet = pd.read_excel(file_path, sheet_name=0)

# Convert all column names to string type for consistency
df_first_sheet.columns = df_first_sheet.columns.astype(str)

# Define the functions to calculate each vegetation index
def calculate_indices(df):
    indices_df = pd.DataFrame()
    indices_df['Blog(1/R737)'] = np.log(1 / df['737'])
    indices_df['TCARI2'] = 3 * ((df['750'] - df['705']) - 0.2 * (df['750'] - df['550']) * (df['750'] / df['705']))
    indices_df['TCARI'] = 3 * ((df['700'] - df['670']) - 0.2 * (df['700'] - df['550']) * (df['700'] / df['670']))
    indices_df['MCARI_OSAVI'] = (((df['700'] - df['670']) - 0.2 * (df['700'] - df['550'])) * (df['700'] / df['670'])) / ((1 + 0.16) * (df['800'] - df['670']) / (df['800'] + df['670'] + 0.16))
    indices_df['EVI'] = 2.5 * ((df['800'] - df['670']) / (df['800'] - (6 * df['670']) - (7.5 * df['475']) + 1))
    indices_df['MCARI'] = ((df['700'] - df['670']) - 0.2 * (df['700'] - df['550'])) * (df['700'] / df['670'])
    indices_df['TCARI2_OSAVI2'] = (3 * ((df['750'] - df['705']) - 0.2 * (df['750'] - df['550']) * (df['750'] / df['705']))) / ((1 + 0.16) * (df['750'] - df['705']) / (df['750'] + df['705'] + 0.16))
    indices_df['DDn'] = 2 * df['710'] - df['660'] - df['760']
    indices_df['MSAVI'] = 0.5 * (2 * df['800'] + 1 - np.sqrt((2 * df['800'] + 1) ** 2 - 8 * (df['800'] - df['670'])))
    indices_df['RDVI'] = (df['800'] - df['670']) / np.sqrt(df['800'] + df['670'])
    indices_df['Sum_Dr1A'] = df.loc[:, '625':'795'].diff(axis=1).sum(axis=1)
    indices_df['TCARI_OSAVI'] = (3 * ((df['700'] - df['670']) - 0.2 * (df['700'] - df['550']) * (df['700'] / df['670']))) / ((1 + 0.16) * (df['800'] - df['670']) / (df['800'] + df['670'] + 0.16))
    indices_df['MCARI2'] = ((df['750'] - df['705']) - 0.2 * (df['750'] - df['550'])) * (df['750'] / df['705'])
    indices_df['NDVI3A'] = (df['682'] - df['553']) / (df['682'] + df['553'])
    indices_df['MTCI'] = (df['754'] - df['709']) / (df['709'] - df['681'])
    indices_df['MCARI2_OSAVI3'] = (((df['750'] - df['705']) - 0.2 * (df['750'] - df['550'])) * (df['750'] / df['705'])) / (1 + 0.16) * (df['750'] - df['705']) / (df['750'] + df['705'] + 0.16)
    indices_df['OSAVI'] = (1 + 0.16) * (df['800'] - df['670']) / (df['800'] + df['670'] + 0.16)
    indices_df['OSAVI2'] = (1 + 0.16) * (df['750'] - df['705']) / (df['750'] + df['705'] + 0.16)
    indices_df['NDVI'] = (df['800'] - df['670']) / (df['800'] + df['670'])
    indices_df['NVI'] = (df['762'] - df['640']) / (df['762'] - df['732'])
    return indices_df

# Filter the DataFrame to include only the columns representing the bands
df_bands = df_first_sheet.loc[:, '325':'1075']

# Calculate the vegetation indices
indices_df = calculate_indices(df_bands)

# Calculate the correlation with 'Chl'
correlation_with_chl = indices_df.corrwith(df_first_sheet['Chl'])
abs_correlation_with_chl = correlation_with_chl.abs()
# 将绝对值赋值给一个新的数据框
df_absolute_correlation = pd.DataFrame({'Vegetation_Indices': abs_correlation_with_chl.index, 'Absolute_Correlation_with_Chl': abs_correlation_with_chl.values})
print(correlation_with_chl)
