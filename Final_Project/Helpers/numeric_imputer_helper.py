from sklearn.impute import SimpleImputer

def categorize_features(df, low_threshold=10, medium_threshold=100):
    categories = {
        'binary': [],
        'low_cardinality': [],
        'medium_cardinality': [],
        'high_cardinality': []
    }
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            categories['binary'].append(col)
        elif unique_vals <= low_threshold:
            categories['low_cardinality'].append(col)
        elif unique_vals <= medium_threshold:
            categories['medium_cardinality'].append(col)
        else:
            categories['high_cardinality'].append(col)
    return categories

def impute_data(df, categories):
    df_imputed = df.copy()
    # Mode imputation for binary and low cardinality features
    mode_imputer = SimpleImputer(strategy='most_frequent')
    mode_cols = categories['binary'] + categories['low_cardinality']
    df_imputed[mode_cols] = mode_imputer.fit_transform(df_imputed[mode_cols])
    
    # Median imputation for medium cardinality features
    median_imputer = SimpleImputer(strategy='median')
    df_imputed[categories['medium_cardinality']] = median_imputer.fit_transform(df_imputed[categories['medium_cardinality']])
    
    # Mean imputation for high cardinality features
    mean_imputer = SimpleImputer(strategy='mean')
    df_imputed[categories['high_cardinality']] = mean_imputer.fit_transform(df_imputed[categories['high_cardinality']])
    
    return df_imputed

def impute_categorical_features_test(df_test, imputation_strategies):
    df_imputed = df_test.copy()
    for col, strategy_info in imputation_strategies.items():
        strategy = strategy_info['imputation_strategy']
        if strategy == 'mode':
            # Use the mode from training data
            mode_value = df_imputed[col].mode(dropna=True)[0]
            df_imputed[col].fillna(mode_value, inplace=True)
        elif strategy == 'constant':
            df_imputed[col].fillna('Unknown', inplace=True)
        elif strategy == 'none':
            pass  # No imputation needed
        else:
            raise ValueError(f"Unknown imputation strategy '{strategy}' for feature '{col}'")
    return df_imputed