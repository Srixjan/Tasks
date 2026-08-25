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

def build_borough_lookup() -> pd.DataFrame:
    borough_data = {
    "neighbourhood_group": df["neighbourhood_group"].unique(),
    "borough_population_milltions": [2.6, 1.6, 2.3, 0.5, 1.4],
    "is_manhattan_adjacent": ['True', 'False', 'True', 'False', 'True']
    }

    return borough_data

borough_data = build_borough_lookup()
borough_info = pd.DataFrame(borough_data)
borough_info

def merge_listings(df: pd.DataFrame, lookup: pd.DataFrame, how:str) -> pd.DataFrame:
    return df.merge(lookup, on="neighbourhood_group", how=how)

testing = merge_listings(df, borough_info, "left")
testing

testing.shape[0] == df.shape[0]

borough_info_2 = borough_info.copy()
borough_info_2 = borough_info_2.drop([0, 1])

testing_2 = merge_listings(df, borough_info_2, "left")
testing_2.shape[0] == df.shape[0]

print(groupby_multi(df))
print(groupby_summary(df))