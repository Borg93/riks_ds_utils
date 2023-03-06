import pandas as pd


def aggregate_mean(df, column):
    return df.groupby("class")[column].mean().to_dict()


if __name__ == "__main__":
    print("hello world")
