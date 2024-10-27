import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce

def categorize_categorical_features_for_encoding(df, low_cardinality_threshold=10, high_cardinality_threshold=100):
    """
    Categorizes categorical features into encoding strategies based on their unique value counts.

    Parameters:
    df (pd.DataFrame): DataFrame containing categorical features.
    low_cardinality_threshold (int): Maximum number of unique values for a feature to be considered low cardinality.
    high_cardinality_threshold (int): Minimum number of unique values for a feature to be considered high cardinality.

    Returns:
    encoding_categories (dict): Dictionary with lists of feature names for each encoding strategy.
    """
    encoding_categories = {
        'binary': [],
        'one_hot': [],
        'frequency': [],
        'target': []
    }
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            encoding_categories['binary'].append(col)
        elif unique_vals <= low_cardinality_threshold:
            encoding_categories['one_hot'].append(col)
        elif unique_vals <= high_cardinality_threshold:
            encoding_categories['frequency'].append(col)
        else:
            encoding_categories['target'].append(col)
    return encoding_categories

def encode_categorical_features(df, encoding_categories, target=None):
    df_encoded = df.copy()
    encoders = {}

    # Binary Encoding using OrdinalEncoder
    if encoding_categories['binary']:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_encoded[encoding_categories['binary']] = ordinal_encoder.fit_transform(df_encoded[encoding_categories['binary']])
        encoders['binary'] = ordinal_encoder

    # One-Hot Encoding
    if encoding_categories['one_hot']:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        one_hot_encoded = one_hot_encoder.fit_transform(df_encoded[encoding_categories['one_hot']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(encoding_categories['one_hot']))
        one_hot_encoded_df.index = df_encoded.index
        df_encoded = df_encoded.drop(columns=encoding_categories['one_hot'])
        df_encoded = pd.concat([df_encoded, one_hot_encoded_df], axis=1)
        encoders['one_hot'] = one_hot_encoder

    # Frequency Encoding
    if encoding_categories['frequency']:
        frequency_maps = {}
        for col in encoding_categories['frequency']:
            freq = df_encoded[col].value_counts()
            df_encoded[col] = df_encoded[col].map(freq)
            frequency_maps[col] = freq
        encoders['frequency'] = frequency_maps

    # Target Encoding
    if encoding_categories['target']:
        if target is None:
            raise ValueError("Target variable must be provided for target encoding.")
        target_encoder = ce.TargetEncoder(cols=encoding_categories['target'])
        df_encoded[encoding_categories['target']] = target_encoder.fit_transform(df_encoded[encoding_categories['target']], target)
        encoders['target'] = target_encoder

    return df_encoded, encoders

def encode_categorical_features_test(df, encoders, encoding_categories):
    df_encoded = df.copy()

    # Binary Encoding using OrdinalEncoder
    if 'binary' in encoders and encoding_categories['binary']:
        ordinal_encoder = encoders['binary']
        df_encoded[encoding_categories['binary']] = ordinal_encoder.transform(df_encoded[encoding_categories['binary']])
    
    # One-Hot Encoding
    if 'one_hot' in encoders and encoding_categories['one_hot']:
        one_hot_encoder = encoders['one_hot']
        one_hot_encoded = one_hot_encoder.transform(df_encoded[encoding_categories['one_hot']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(encoding_categories['one_hot']))
        one_hot_encoded_df.index = df_encoded.index
        df_encoded = df_encoded.drop(columns=encoding_categories['one_hot'])
        df_encoded = pd.concat([df_encoded, one_hot_encoded_df], axis=1)
    
    # Frequency Encoding
    if 'frequency' in encoders and encoding_categories['frequency']:
        frequency_maps = encoders['frequency']
        for col in encoding_categories['frequency']:
            freq_map = frequency_maps[col]
            df_encoded[col] = df_encoded[col].map(freq_map)
            # Handle unseen categories by filling NaN values
            df_encoded[col] = df_encoded[col].fillna(0)  # Or fill with mean frequency
    
    # Target Encoding
    if 'target' in encoders and encoding_categories['target']:
        target_encoder = encoders['target']
        df_encoded[encoding_categories['target']] = target_encoder.transform(df_encoded[encoding_categories['target']])

    return df_encoded
