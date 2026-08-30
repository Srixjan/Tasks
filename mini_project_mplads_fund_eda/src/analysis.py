import pandas as pd

def grouping_by_mps(df: pd.DataFrame, grouping_column: str, amount_column: str) -> pd.DataFrame:
    return df.groupby(grouping_column).agg(
        total_sanction_amount=(amount_column, "sum"),
        sanction_work_count=(amount_column, "count")
    )