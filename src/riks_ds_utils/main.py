# import pandas as pd
from typing import TypeVar


DataFrame = TypeVar("pandas.core.frame.DataFrame")


def aggregate_mean(df: DataFrame, column: str) -> dict:
    """Computes and returns the mean value of a column

    Args:
        df (DataFrame): A pandas dataframe
        column (str): Column name from the dataframe

    Returns:
        dict: Mean value of each column as a dict.
    """
    return df.groupby("class")[column].mean().to_dict()


if __name__ == "__main__":
    print("hello world")

    # Usage:

    # How to use the function:

    # ```python
    # from riks_ds_utils.main import aggregate_mean

    # mean_dict = aggregate_mean(df, column)

    # ```
