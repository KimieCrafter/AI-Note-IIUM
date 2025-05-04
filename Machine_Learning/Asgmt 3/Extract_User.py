import pandas as pd
import random

# Load the test CSV
df = pd.read_csv("Machine_Learning/Asgmt 3/test.csv")

# Show unique subject IDs
print("Available subject IDs:", df['subject'].unique())

user_id = 2  # change this to any user ID to sample from

# Filter data for that user
user_data = df[df['subject'] == user_id]

# Ensure we have rows
if user_data.empty:
    print(f"No data found for subject ID {user_id}. Please try another.")
else:
    # Randomly select one row from the user's data
    random_row = user_data.sample(n=25)
    
    random_row = random_row.drop(columns=['subject']) # add 'Activity' if needed
    
    # Save the random row to Excel
    random_row.to_excel("Machine_Learning/Asgmt 3/user.xlsx", index=False)
    
    print(f"Extracted a random data row for subject {user_id} and saved to 'random_user_sample.xlsx'")
