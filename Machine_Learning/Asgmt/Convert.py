import pandas as pd

# Load the Excel file
excel_data = pd.read_excel('/mnt/data/SolarPowerGenerationKaggle_MissingData.xlsx')

# Save as CSV
excel_data.to_csv('/mnt/data/converted_data.csv', index=False)
