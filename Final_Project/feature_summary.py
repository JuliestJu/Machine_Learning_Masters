import pandas as pd
import numpy as np

def export_feature_summary(data: pd.DataFrame, output_csv: str, unique_threshold: int = 15):
    """
    Processes the dataset to extract feature summaries and exports them to a CSV file.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - output_csv (str): The file path for the output CSV.
    - unique_threshold (int, optional): Maximum number of unique values to display. Defaults to 15.
    """
    # Separate numeric and categorical features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object', 'category']).columns

    summary_rows = []

    # Total number of rows in the dataset
    total_rows = data.shape[0]

    # Helper function to process each feature
    def process_feature(feature, feature_type):
        unique_values = data[feature].dropna().unique()
        unique_count = len(unique_values)
        total_non_null = data[feature].count()
        missing_count = total_rows - total_non_null

        # Convert unique values to string, handling NaN
        if unique_count <= unique_threshold:
            unique_values_str = ', '.join([str(val) if pd.notna(val) else 'nan' for val in unique_values])
        else:
            unique_values_str = f"More than {unique_threshold} unique values"

        summary_rows.append({
            'Feature Name': feature,
            'Unique Value Count': unique_count,
            'Unique Values': unique_values_str,
            'Total Row Count': total_non_null,
            'Missing Value Count': missing_count,
            'Feature Type': feature_type
        })

    # Process Numeric Features
    for feature in numeric_features:
        process_feature(feature, 'Numeric')

    # Process Categorical Features
    for feature in categorical_features:
        process_feature(feature, 'Categorical')

    # Create Summary DataFrame
    summary_df = pd.DataFrame(summary_rows, columns=[
        'Feature Name',
        'Unique Value Count',
        'Unique Values',
        'Total Row Count',
        'Missing Value Count',
        'Feature Type'
    ])

    # Export to CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"Feature summary exported to {output_csv}")

# Example usage:
# df = pd.read_csv('your_dataset.csv')
# export_feature_summary(df, 'feature_summary.csv')
