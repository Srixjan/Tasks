import pandas as pd

def grouping_by_mps(df: pd.DataFrame, grouping_column: str, amount_column: str) -> pd.DataFrame:
    return df.groupby(grouping_column).agg(
        total_sanction_amount=(amount_column, "sum"),
        sanction_work_count=(amount_column, "count")
    )

def run_task6_groupby_analysis(df: pd.DataFrame, ws: pd.DataFrame, wc: pd.DataFrame):
    state_summary = df[df["state"] != ""].groupby("state").agg(
        avg_utilization_rate=("utilization_rate", "mean"),
        avg_sanctioned_backlog=("sanctioned_backlog", "mean")
    ).sort_values(by="avg_utilization_rate", ascending=False)

    cutoff_value = 1.47e08
    filtered_df = df[df["allocated_amount"] > cutoff_value]
    top_10_mps = filtered_df.sort_values(by="utilization_rate", ascending=False).head(10)
    bottom_10_mps = filtered_df.sort_values(by="utilization_rate", ascending=True).head(10)

    sanction_by_category = ws.groupby("work_category").agg(
        total_sanction_amount=("sanction_amount", "sum")
    )
    disbursed_by_category = wc.groupby("work_category").agg(
        total_disbursed_amount=("amount_disbursed", "sum")
    )
    check = pd.merge(sanction_by_category, disbursed_by_category, on="work_category", how="left")
    check["gap"] = check["total_sanction_amount"] - check["total_disbursed_amount"]
    check = check.sort_values(by="gap", ascending=False)

    return state_summary, top_10_mps, bottom_10_mps, check