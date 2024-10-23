import os

# Define the function to split the dataset
def split_dataset_by_missing_and_type(df):
    # Filter and sort data based on the conditions provided

    # 1) Numeric features with zero missing values, sorted by Unique Values
    numeric_no_missing = df[(df['Feature Type'] == 'Numeric') & (df['Missing Value Count'] == 0)].sort_values(by='Unique Values')

    # 2) Categorical features with zero missing values, sorted by Unique Values
    categorical_no_missing = df[(df['Feature Type'] == 'Categorical') & (df['Missing Value Count'] == 0)].sort_values(by='Unique Values')

    # 3) Numeric features with non-zero missing values, sorted by Unique Values
    numeric_with_missing = df[(df['Feature Type'] == 'Numeric') & (df['Missing Value Count'] > 0)].sort_values(by='Unique Values')

    # 4) Categorical features with non-zero missing values, sorted by Unique Values
    categorical_with_missing = df[(df['Feature Type'] == 'Categorical') & (df['Missing Value Count'] > 0)].sort_values(by='Unique Values')

    # Ensure the 'splits' directory exists before saving files
    splits_dir = 'splits'
    os.makedirs(splits_dir, exist_ok=True)

    # Save the filtered and sorted datasets to CSV files in the 'splits' folder
    numeric_no_missing.to_csv(os.path.join(splits_dir, 'numeric_no_missing.csv'), index=False)
    categorical_no_missing.to_csv(os.path.join(splits_dir, 'categorical_no_missing.csv'), index=False)
    numeric_with_missing.to_csv(os.path.join(splits_dir, 'numeric_with_missing.csv'), index=False)
    categorical_with_missing.to_csv(os.path.join(splits_dir, 'categorical_with_missing.csv'), index=False)

    return "Files saved successfully."

