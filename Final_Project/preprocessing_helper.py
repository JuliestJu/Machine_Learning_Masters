import pandas as pd

def split_dataset(df):
    """
    Function to split a dataset into numerical and categorical features based on both data type and unique values.
    
    Args:
    df (pd.DataFrame): The input dataset to process.
    unique_value_threshold (int): Maximum number of unique values to consider a feature as categorical.
    
    Returns:
    tuple: Two DataFrames (numerical_features, categorical_features)
    """
    # Select features based on data types
    numeric_features = df.select_dtypes(include=['number'])
    categorical_features = df.select_dtypes(include=['object'])

    return numeric_features, categorical_features

def split_dataframe_by_missing_values(df: pd.DataFrame):
    """
    Splits the input DataFrame into two DataFrames:
    1. DataFrame with columns that have missing values.
    2. DataFrame with columns that do not have missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
        - DataFrame with missing values.
        - DataFrame without missing values.
    """
    # DataFrame with columns that have missing values
    df_with_missing = df.loc[:, df.isnull().any()]

    # DataFrame with columns that do not have missing values
    df_without_missing = df.loc[:, ~df.isnull().any()]

    return df_with_missing, df_without_missing

import pandas as pd

def get_imputation_feature_lists_from_dataset(df, missing_threshold=0.3):
    """
    Returns lists of features for different imputation strategies:
    - Low unique value count (<10): Mode imputation
    - Moderate unique value count (10-40): Median imputation
    - High unique value count (>40): Mean imputation
    - High missing percentage (configurable threshold): Special handling or removal
    
    Parameters:
    df (pd.DataFrame): Original dataset containing numerical features
    missing_threshold (float): Threshold for missing percentage to flag features for special handling (default is 0.3)
    
    Returns:
    dict: Dictionary with lists of feature names for each strategy
    """
    # Calculate the number of unique values for each feature
    unique_value_counts = df.nunique()

    # Calculate the total number of rows in the dataset
    total_row_count = len(df)

    # Calculate the number of missing values for each feature
    missing_value_counts = df.isnull().sum()

    # Calculate the percentage of missing values for each feature
    missing_percentage = missing_value_counts / total_row_count

    # Low unique value count (<10) -> Mode imputation
    low_unique_features = unique_value_counts[unique_value_counts < 10].index.tolist()

    # Moderate unique value count (10-40) -> Median imputation
    moderate_unique_features = unique_value_counts[(unique_value_counts >= 10) & (unique_value_counts <= 40)].index.tolist()

    # High unique value count (>40) -> Mean imputation
    high_unique_features = unique_value_counts[unique_value_counts > 40].index.tolist()

    # High missing percentage (using the threshold passed as a parameter) -> Special handling
    high_missing_features = missing_percentage[missing_percentage > missing_threshold].index.tolist()

    # Return the feature lists
    return {
        "low_unique_features": low_unique_features,
        "moderate_unique_features": moderate_unique_features,
        "high_unique_features": high_unique_features,
        "high_missing_features": high_missing_features
    }


