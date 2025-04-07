import pandas as pd

# Load the Excel file
excel_data = pd.read_excel('Machine_Learning\Asgmt 2\heart_attack_prediction_dataset_Original.xlsx')

# Save as CSV
excel_data.to_csv('heart_attack_prediction', index=False)
