import pandas as pd

# Step 1: Load the dataset
file_path = r"C:\Users\HP-PC\Desktop\EXPRESSO_CHURN_PREDICTION\data\Expresso_churn_dataset.csv"  # dataset's path
df = pd.read_csv(file_path,encoding='latin1')

# Step 2: View dataset information
print("Dataset Info:")
print(df.info())

# Step 3: Check for rows with missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# Step 4: Drop rows with missing values in specific columns
columns_to_check = ['REVENUE',	'CHURN' ]  #column names
df = df.dropna(subset=columns_to_check)

# Step 5: Fill remaining missing values
for column in df.columns:
    if df[column].dtype == 'object':  # Categorical column
        mode_value = df[column].mode()[0]  # Get the most frequent value
        df[column] = df[column].fillna(mode_value)
    else:  # Numerical column
        mean_value = df[column].mean()  # Calculate the mean
        df[column] = df[column].fillna(mean_value)

# Step 6: Save the cleaned dataset
cleaned_file_path = "cleaned_dataset.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")
