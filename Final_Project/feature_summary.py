import pandas as pd

def export_feature_summary(data: pd.DataFrame, output_csv: str, unique_threshold: int = 15, ascending=True):
    """
    Processes the dataset to extract feature summaries, sorts them by unique value count, 
    and exports the summary to a CSV file.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - output_csv (str): The file path for the output CSV.
    - unique_threshold (int, optional): Maximum number of unique values to display. Defaults to 15.
    - ascending (bool, optional): Sort order for unique value count. Defaults to True.
    """
    summary_rows = []

    # Total number of rows in the dataset
    total_rows = data.shape[0]

    # Helper function to process each feature
    def process_feature(feature):
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
        })

    # Process all features (no distinction between numeric and categorical)
    for feature in data.columns:
        process_feature(feature)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows, columns=[
        'Feature Name',
        'Unique Value Count',
        'Unique Values',
        'Total Row Count',
        'Missing Value Count'
    ])

    # Sort the summary DataFrame by 'Unique Value Count'
    summary_df = summary_df.sort_values(by='Unique Value Count', ascending=ascending).reset_index(drop=True)

    # Export to CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"Feature summary exported to {output_csv}")

# Example usage:
# df = pd.read_csv('your_dataset.csv')
# export_feature_summary(df, 'feature_summary.csv')