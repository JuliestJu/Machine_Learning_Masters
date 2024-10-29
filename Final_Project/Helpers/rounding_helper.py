import numpy as np
from math import gcd
from functools import reduce

def compute_gcd_of_list(num_list):
    num_list = [int(round(num)) for num in num_list if not np.isnan(num)]
    if len(num_list) < 2:
        return None
    return reduce(gcd, num_list)

def find_rounding_features(df, min_gcd=2, min_unique_values=5):
    rounding_features = {}
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < min_unique_values:
            continue
        unique_vals_sorted = np.sort(unique_vals)
        differences = np.diff(unique_vals_sorted)
        differences = differences[differences > 0]
        if len(differences) < 1:
            continue
        gcd_of_diffs = compute_gcd_of_list(differences)
        if gcd_of_diffs and gcd_of_diffs >= min_gcd:
            diffs_mod_gcd = differences % gcd_of_diffs
            if np.sum(diffs_mod_gcd) == 0:
                rounding_features[col] = gcd_of_diffs
    return rounding_features

def round_detected_features(df, rounding_features):
    df_rounded = df.copy()
    for feature, multiple in rounding_features.items():
        df_rounded[feature] = df_rounded[feature].apply(lambda x: round_to_nearest_multiple(x, multiple))
    return df_rounded

def round_to_nearest_multiple(x, multiple):
    """
    Rounds the given number x to the nearest multiple of the specified value.

    Parameters:
    x (float or int): The number to be rounded.
    multiple (int): The multiple to which x should be rounded.

    Returns:
    int or float: The number rounded to the nearest multiple.
    """
    return multiple * round(x / multiple)

# print("Features that may require rounding:")
# for feature, multiple in rounding_features.items():
#     print(f"{feature}: Round to nearest multiple of {multiple}")
