import os
import pandas as pd
import numpy as np

class FileValidationError(Exception):
    pass

class FileMissingValueError(FileValidationError):
    pass

def load_results(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileValidationError("File not found!!")  
    elif os.path.getsize(filepath) == 0:
        raise FileMissingValueError("File exists but contains no value or data")
    else:
        return pd.read_csv(filepath)
    
def handle_missing_values(df: pd.DataFrame, strategy: str="drop") -> pd.DataFrame:
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill":
        df = df.fillna(df.select_dtypes(include="number").mean())
    else: 
        print("Wrong Message!!")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def fix_dtypes(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    df[column] = pd.to_numeric(df[column], errors="coerce").astype(target_type)
    return df

def clean_string_column(df: pd.DataFrame, columns: str) -> pd.DataFrame:
    df[columns] = df[columns].astype(str).str.strip().str.lower()
    return df

if __name__ == "__main__":
    df = load_results("../data/pandas_p1/results.csv")
    df.replace("\\N", np.nan, inplace=True)
    df = remove_duplicates(df)
    df = handle_missing_values(df, "drop")
    df = fix_dtypes(df, "points", "float64")
    # df = clean_string_column(df)
