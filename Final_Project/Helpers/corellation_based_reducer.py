import pandas as pd
import numpy as np

def correlation_based_elimination(df, threshold=0.8):
    
    corr_matrix = df.corr().abs()
    
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    
    reduced_df = df.drop(columns=to_drop)
    
    return reduced_df, to_drop

# Usage:
# reduced_df, dropped_columns = correlation_based_elimination(your_dataframe)