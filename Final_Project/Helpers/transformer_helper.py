import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

def categorize_numeric_features_by_skewness_and_variance(df, skew_threshold=0.5, unique_value_threshold=10, variance_threshold=0.01):
    """
    Categorizes numeric features based on skewness, unique value counts, and variance.

    Parameters:
    df (pd.DataFrame): DataFrame containing numeric features.
    skew_threshold (float): Threshold for skewness to decide if a feature is highly skewed.
    unique_value_threshold (int): Threshold to identify discrete features.
    variance_threshold (float): Threshold for variance to categorize features.

    Returns:
    categories (dict): Dictionary with lists of feature names for each transformation category.
    """
    categories = {
        'no_transform': [],
        'power_transform': [],
        'log_transform': [],
        'standard_scaler': [],
        'minmax_scaler': [],
        'binary': [],
        'discrete': []
    }

    for col in df.columns:
        unique_vals = df[col].nunique()
        skewness = df[col].skew()
        variance = df[col].var()

        if unique_vals == 2:
            categories['binary'].append(col)
        elif unique_vals <= unique_value_threshold:
            categories['discrete'].append(col)
        elif abs(skewness) >= skew_threshold:
            if (df[col] > 0).all():
                categories['log_transform'].append(col)
            else:
                categories['power_transform'].append(col)
        elif variance >= variance_threshold:
            categories['standard_scaler'].append(col)
        else:
            categories['minmax_scaler'].append(col)

    return categories

def transform_numeric_features(df, categories):
    """
    Applies transformations to numeric features based on their categories.

    Parameters:
    df (pd.DataFrame): DataFrame containing numeric features.
    categories (dict): Dictionary with lists of feature names for each transformation category.

    Returns:
    df_transformed (pd.DataFrame): DataFrame with transformed numeric features.
    transformers (dict): Dictionary of fitted transformers for each category.
    """
    df_transformed = df.copy()
    transformers = {}

    # Power Transformation
    if categories['power_transform']:
        pt = PowerTransformer(method='yeo-johnson')
        df_transformed[categories['power_transform']] = pt.fit_transform(df_transformed[categories['power_transform']])
        transformers['power_transformer'] = pt

    # Log Transformation
    if categories['log_transform']:
        df_transformed[categories['log_transform']] = np.log1p(df_transformed[categories['log_transform']])
        transformers['log_transform'] = 'Applied log1p'

    # Standard Scaling for high variance features
    if categories['standard_scaler']:
        scaler = StandardScaler()
        df_transformed[categories['standard_scaler']] = scaler.fit_transform(df_transformed[categories['standard_scaler']])
        transformers['standard_scaler'] = scaler

    # Min-Max Scaling for low variance features
    if categories['minmax_scaler']:
        minmax_scaler = MinMaxScaler()
        df_transformed[categories['minmax_scaler']] = minmax_scaler.fit_transform(df_transformed[categories['minmax_scaler']])
        transformers['minmax_scaler'] = minmax_scaler

    return df_transformed, transformers

def transform_numeric_features_test(df_test, transformers, categories):
    """
    Applies the same transformations to test data using fitted transformers from training data.

    Parameters:
    df_test (pd.DataFrame): Test DataFrame containing numeric features.
    transformers (dict): Dictionary of fitted transformers from training data.
    categories (dict): Dictionary with lists of feature names for each transformation category.

    Returns:
    df_transformed (pd.DataFrame): Transformed test DataFrame.
    """
    df_transformed = df_test.copy()

    # Power Transformation
    if 'power_transformer' in transformers:
        pt = transformers['power_transformer']
        df_transformed[categories['power_transform']] = pt.transform(df_transformed[categories['power_transform']])

    # Log Transformation
    if 'log_transform' in transformers:
        df_transformed[categories['log_transform']] = np.log1p(df_transformed[categories['log_transform']])

    # Standard Scaling
    if 'standard_scaler' in transformers:
        scaler = transformers['standard_scaler']
        df_transformed[categories['standard_scaler']] = scaler.transform(df_transformed[categories['standard_scaler']])

    # Min-Max Scaling
    if 'minmax_scaler' in transformers:
        minmax_scaler = transformers['minmax_scaler']
        df_transformed[categories['minmax_scaler']] = minmax_scaler.transform(df_transformed[categories['minmax_scaler']])

    return df_transformed
