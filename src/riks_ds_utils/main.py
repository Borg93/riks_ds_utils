import pandas as pd


def aggregate_mean(df :pd.DataFrame, column:str) -> dict:
    """ Computes and returns the mean value of a column 

    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

    Arguments: 
        df: A pandas dataframe
        column: Column name from the dataframe
    
    Returns: 
       dict:  Mean value of each column as a dict.
       
    Usage:

    How to use the function:

    ```python
    from riks_ds_utils.main import aggregate_mean

    mean_dict = aggregate_mean(df, column)
    
    ```
    """
    return df.groupby("class")[column].mean().to_dict()


if __name__ == "__main__":
    print("hello world")
