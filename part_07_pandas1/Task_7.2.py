import pandas as pd
import os
print(os.getcwd())

class FileValidationError(Exception):
    """Incase file not found or doesnt exist!"""
    pass
class FileMissingValueError(FileValidationError):
    """If file exists but contains no value at all!"""
    pass

def load_results(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileValidationError("File not found!!")  
    
    elif os.path.getsize(filepath) == 0:
        raise FileMissingValueError("File exists but contain no value or data")

    else:
        return pd.read_csv(filepath)
        
def filter_by_points_range(df: pd.DataFrame, min_points: int, max_points: int) -> pd.DataFrame:
    return df.query("points >= @min_points & points <= @max_points")

def filter_by_grid_and_position(df: pd.DataFrame, grid_positions: list[int], finish_positions: list[str]) -> pd.DataFrame:
    return df[df["grid"].isin(grid_positions) & df["position"].isin(finish_positions)]

def get_top_constructors(df: pd.DataFrame , n: int=5) -> list:
    return df["constructorId"].value_counts().head(n).index.tolist()

def get_points_outliers(df: pd.DataFrame, std_threshold: int=3) -> pd.DataFrame:
    mean = df["points"].mean()
    std = df["points"].std()
    cutoff = std_threshold * std
    return df.loc[(df["points"] > (mean + cutoff)) | (df["points"] < (mean - cutoff))]



if __name__ == "__main__":

    df = load_results("../data/pandas_p1/results.csv")

    print("--- Full DataFrame ---")
    print(df)
    print("\n--- DataFrame Shape ---")
    print(df.shape)
    
    print("\n--- Filtered by Points Range (10 to 25) ---")
    print(filter_by_points_range(df, min_points=10, max_points=25))
    
    print("\n--- Filtered by Grid (1) and Position ('1') ---")
    print(filter_by_grid_and_position(df, grid_positions=[1], finish_positions=["1"]))
    
    print("\n--- Top 5 Constructors ---")
    print(get_top_constructors(df, 5))
    
    print("\n--- Points Outliers (Threshold: 5) ---")
    print(get_points_outliers(df, std_threshold=5))