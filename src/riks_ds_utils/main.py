import pandas as pd


def aggregate_mean(df, column):
    """ Computes and returns the mean value of a column 

    Args: 
        df (pd.Dataframe): A pandas dataframe
        column (string): Column name from the dataframe
    
    Returns: 
        Mean value of each column as a dict.
    """
    return df.groupby("class")[column].mean().to_dict()


if __name__ == "__main__":
    print("hello world")
