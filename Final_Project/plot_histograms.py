import matplotlib.pyplot as plt

# Refactored function to plot histograms of selected columns and calculate variance
def plot_histograms_with_variance(df, columns, bins=30):
    num_columns = len(columns)
    rows = (num_columns // 6) + 1  # To create a grid of plots
    plt.figure(figsize=(20, rows * 5))
    
    for i, col in enumerate(columns, 1):
        plt.subplot(rows, 6, i)
        df[col].hist(bins=bins)
        variance = df[col].var()
        plt.title(f'Histogram of {col}\nVariance: {variance:.2f}')
        plt.tight_layout()

# Example usage:
# plot_histograms_with_variance(df, selected_columns)
