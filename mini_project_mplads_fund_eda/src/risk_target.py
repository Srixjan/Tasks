import pandas as pd

class MissingColumnError(Exception):
    """Must be triggered if utilization rate column is missing"""
    pass

def defined_completion_risk(df: pd.DataFrame, method: str, threshold: int=float, group_col: str = None):
    df = df.copy()
    if "utilization_rate" not in df.columns:
        raise MissingColumnError("Column utilization rate doesn't exist in the dataframe!")

    if (method == "threshold"):
        df["at_risk"] = (df["utilization_rate"] <= threshold).astype(int)

    elif(method == "relative"):
        group_col = "state"
        group_medians = df.groupby(group_col)["utilization_rate"].transform("median")
        df["at_risk"] = (df["utilization_rate"] <= group_medians).astype(int)
        
    else:
        raise ValueError(f"Unknown method '{method}'. Choose either 'threshold' or 'relative'.")
        
    return df

