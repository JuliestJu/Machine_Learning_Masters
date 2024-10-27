from sklearn.impute import SimpleImputer

def categorize_categorical_features(df, low_threshold=10, medium_threshold=100):
    """
    Categorizes categorical features into binary, low_cardinality, medium_cardinality, and high_cardinality.

    Parameters:
    df (pd.DataFrame): DataFrame containing categorical features.
    low_threshold (int): Threshold to distinguish between low and medium cardinality features.
    medium_threshold (int): Threshold to distinguish between medium and high cardinality features.

    Returns:
    categories (dict): Dictionary with lists of feature names for each category.
    """
    categories = {
        'binary': [],
        'low_cardinality': [],
        'medium_cardinality': [],
        'high_cardinality': []
    }
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=False)  # Include NaN in unique count
        if unique_vals == 2:
            categories['binary'].append(col)
        elif unique_vals <= low_threshold:
            categories['low_cardinality'].append(col)
        elif unique_vals <= medium_threshold:
            categories['medium_cardinality'].append(col)
        else:
            categories['high_cardinality'].append(col)
    return categories

def impute_categorical_data(df, categories):
    """
    Imputes missing values in categorical features based on their categories.

    Parameters:
    df (pd.DataFrame): DataFrame containing categorical features.
    categories (dict): Dictionary with lists of feature names for each category.

    Returns:
    df_imputed (pd.DataFrame): DataFrame with imputed categorical features.
    imputers (dict): Dictionary of fitted imputers for each category.
    """
    df_imputed = df.copy()
    imputers = {}

    # Mode imputation for binary and low cardinality features
    mode_cols = categories['binary'] + categories['low_cardinality']
    if mode_cols:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[mode_cols] = mode_imputer.fit_transform(df_imputed[mode_cols])
        imputers['mode_imputer'] = mode_imputer

    # Constant imputation with 'Unknown' for medium cardinality features
    medium_cols = categories['medium_cardinality']
    if medium_cols:
        constant_imputer_unknown = SimpleImputer(strategy='constant', fill_value='Unknown')
        df_imputed[medium_cols] = constant_imputer_unknown.fit_transform(df_imputed[medium_cols])
        imputers['constant_imputer_unknown'] = constant_imputer_unknown

    # Constant imputation with 'Missing' for high cardinality features
    high_cols = categories['high_cardinality']
    if high_cols:
        constant_imputer_missing = SimpleImputer(strategy='constant', fill_value='Missing')
        df_imputed[high_cols] = constant_imputer_missing.fit_transform(df_imputed[high_cols])
        imputers['constant_imputer_missing'] = constant_imputer_missing

    return df_imputed, imputers

def impute_categorical_data_test(df, categories, imputers):
    """
    Imputes missing values in test categorical features using fitted imputers from training data.

    Parameters:
    df (pd.DataFrame): Test DataFrame containing categorical features.
    categories (dict): Dictionary with lists of feature names for each category.
    imputers (dict): Dictionary of fitted imputers for each category from training data.

    Returns:
    df_imputed (pd.DataFrame): DataFrame with imputed categorical features.
    """
    df_imputed = df.copy()

    # Mode imputation for binary and low cardinality features
    mode_cols = categories['binary'] + categories['low_cardinality']
    if mode_cols:
        mode_imputer = imputers['mode_imputer']
        df_imputed[mode_cols] = mode_imputer.transform(df_imputed[mode_cols])

    # Constant imputation with 'Unknown' for medium cardinality features
    medium_cols = categories['medium_cardinality']
    if medium_cols:
        constant_imputer_unknown = imputers['constant_imputer_unknown']
        df_imputed[medium_cols] = constant_imputer_unknown.transform(df_imputed[medium_cols])

    # Constant imputation with 'Missing' for high cardinality features
    high_cols = categories['high_cardinality']
    if high_cols:
        constant_imputer_missing = imputers['constant_imputer_missing']
        df_imputed[high_cols] = constant_imputer_missing.transform(df_imputed[high_cols])

    return df_imputed