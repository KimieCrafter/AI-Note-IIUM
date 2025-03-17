import pandas as pd

# Load the Excel file
excel_data = pd.read_excel('SolarPowerGenerationKaggle_MissingData.xlsx')

# Save as CSV
excel_data.to_csv('converted_data.csv', index=False)
