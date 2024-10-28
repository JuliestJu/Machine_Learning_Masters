import pandas as pd
import numpy as np

def correlation_based_elimination(df, threshold=0.8):
    # Step 1: Calculate the correlation matrix
    corr_matrix = df.corr().abs()
    
    # Step 2: Identify highly correlated features
    # Only consider the upper triangle of the correlation matrix to avoid redundancy
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Step 3: Identify columns to drop based on the threshold
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    
    # Step 4: Drop the columns
    reduced_df = df.drop(columns=to_drop)
    
    return reduced_df, to_drop

# Usage:
# reduced_df, dropped_columns = correlation_based_elimination(your_dataframe)
# print("Dropped columns:", dropped_columns)