import pandas as pd

# Load the Excel file
excel_data = pd.read_excel('Solar_Power_Generation.xlsx')

# Save as CSV
excel_data.to_csv('Solar_Power_Generation.csv', index=False)
