import pandas as pd
import numpy as np

df = pd.read_csv("../data/pandas_p1/AB_NYC_2019.csv")

def groupby_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Groups the DataFrame by neighbourhood_group and calculates summary statistics."""
    return df.groupby("neighbourhood_group").agg(
        listing_count=("id", "count"),
        average_price=("price", "mean"),
        average_median=("price", "median"),
        average_availability=("availability_365", "mean")
    ).sort_values("average_price", ascending=False)

def groupby_multi(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["neighbourhood_group", "room_type"]).agg(average_price=("price", "mean")).reset_index()

print(groupby_multi(df))
print(groupby_summary(df))