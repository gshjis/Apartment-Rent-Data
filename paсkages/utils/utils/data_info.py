import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
from prettytable import PrettyTable


def get_numeric_stats(df, exclude_cols=None, n_bins=10):
    """
    Calculates statistics for numeric variables in the given dataframe.

    For numeric variables, it prints:
    - Mean
    - Standard deviation
    - 95% confidence interval for the mean
    - Histogram of distribution with specified number of bins

    Args:
        df (pandas.DataFrame): Input dataframe.
        exclude_cols (list, optional): List of column names to exclude from the analysis. Defaults to None.
        n_bins (int, optional): Number of bins to use for the numeric variable histograms. Defaults to 10.

    Returns:
        None
    """
    # Print statistics for numeric variables
    print("Statistics for numeric variables:")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    table = PrettyTable()
    table.field_names = ["Column", "Mean", "Std. Dev.", "95% CI"]

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        df_len = len(df)
        ci_lower, ci_upper = t.interval(
            0.95, df_len - 1, loc=mean, scale=std / df_len**0.5
        )
        table.add_row(
            [col, f"{mean:.2f}", f"{std:.2f}", f"[{ci_lower:.2f}, {ci_upper:.2f}]"]
        )

    print(table)

    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        df[col].hist(bins=n_bins)
        plt.title(f"Histogram of '{col}'")
        plt.show()


def get_categorical_stats(df, exclude_cols=None):
    """
    Calculates frequency characteristics for categorical variables in the given dataframe.

    For categorical variables, it prints:
    - Frequency distribution
    - Horizontal histogram of distribution with clear separation between columns

    Args:
        df (pandas.DataFrame): Input dataframe.
        exclude_cols (list, optional): List of column names to exclude from the analysis. Defaults to None.

    Returns:
        None
    """
    # Print frequency characteristics for categorical variables
    print("\nFrequency characteristics for categorical variables:")
    cat_cols = df.select_dtypes(include=["object"]).columns
    if exclude_cols:
        cat_cols = [col for col in cat_cols if col not in exclude_cols]

    for col in cat_cols:
        print(f"\nDistribution for column '{col}':")
        print(df[col].value_counts())
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(y=col, data=df, orient="h")
        ax.axhline(y=0, color="k", linestyle="-")
        plt.title(f"Distribution histogram for '{col}'")
        plt.show()
