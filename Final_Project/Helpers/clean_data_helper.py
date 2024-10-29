import pandas as pd

def remove_columns_with_missing_values(df, threshold=0.30):
    """
    This function removes columns with missing values above a specified threshold from a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to clean.
    threshold (float): The proportion of missing values above which columns are removed. Default is 0.30 (30%).

    Returns:
    pd.DataFrame: The cleaned DataFrame with columns removed based on the missing values threshold.
    """
    
    # Calculate the percentage of missing values for each column
    missing_percent = df.isnull().mean()
    
    # Identify columns to drop
    columns_to_drop = missing_percent[missing_percent > threshold].index
    
    # Drop columns from the dataset
    cleaned_data = df.drop(columns=columns_to_drop)
    
    # Output information
    print(f"New dataset shape after column removal: {cleaned_data.shape}")
    
    return cleaned_data

def remove_rows_with_missing_values(df, threshold=0.30):
    """
    This function removes rows from the DataFrame that have more missing values than the specified threshold.

    Args:
    df (pd.DataFrame): The DataFrame to clean.
    threshold (float): The proportion of missing values allowed per row. Default is 0.30 (30%).

    Returns:
    pd.DataFrame: The cleaned DataFrame with rows removed based on the missing values threshold.
    """
    
    # Calculate the minimum number of non-missing values required per row
    row_threshold = int((1 - threshold) * df.shape[1])
    
    # Drop rows that don't meet the row threshold
    cleaned_data = df.dropna(thresh=row_threshold)
    
    # Output information
    print(f"New dataset shape after removing rows with more than {int(threshold * 100)}% missing columns: {cleaned_data.shape}")
    
    return cleaned_data
