import logging
import pandas as pd
import numpy as np
logging.basicConfig(level=logging.INFO)

from src.data_pipeline import FileMissingValueError, FileValidationError

from src.data_pipeline import diagnose_dataframe, load_results

from src.analysis import grouping_by_mps

from src.data_pipeline import (
    clean_dataframe_columns,
    convert_real_datetime,
    drop_duplicate_rows,
    mp_name_clean,
    numeric_conversion,
    replace_strings_with_real_nulls,
    strip_whitespace,
)

if __name__ == "__main__":
    try:
        al = load_results("data/raw/Allocated Limit for Honble MPs.csv")
        wc = load_results("data/raw/Works Completed.csv")
        ws = load_results("data/raw/Works Sanctioned.csv")
    except FileValidationError as e:
        logging.error(f"Loading Failed: {e}")
        raise

    al = clean_dataframe_columns(al)
    wc = clean_dataframe_columns(wc)
    ws = clean_dataframe_columns(ws)

    diagnose_dataframe(al, "Allocated Limit for Honble MPs")
    diagnose_dataframe(wc, "Works Completed")
    diagnose_dataframe(ws, "Works Sanctioned")


    al = numeric_conversion(al, ["allocated_amount"])
    wc = numeric_conversion(wc, ["amount_disbursed"])
    ws = numeric_conversion(ws, ["sanction_amount"])

    al = replace_strings_with_real_nulls(al)
    wc = replace_strings_with_real_nulls(wc)
    ws = replace_strings_with_real_nulls(ws)

    al = strip_whitespace(al)
    wc = strip_whitespace(wc)
    ws = strip_whitespace(ws)

    wc = convert_real_datetime(wc, ["completion_date"])
    ws = convert_real_datetime(ws, ["recommended_date", "sanction_date"])

    al = drop_duplicate_rows(al, "Allocated Limit")
    ws = drop_duplicate_rows(ws, "Works Sanctioned")
    wc = drop_duplicate_rows(wc, "Works Completed")

    al = al.drop(columns=["sr_no"])
    ws = ws.drop(columns=["sr_no"])
    wc = wc.drop(columns=["sr_no", "image"])
 
    al["honble_members_of_parliaments"] = al["honble_members_of_parliaments"].apply(mp_name_clean)
    ws["honble_members_of_parliament"] = ws["honble_members_of_parliament"].apply(mp_name_clean)
    wc["honble_members_of_parliament"] = wc["honble_members_of_parliament"].apply(mp_name_clean)

    al_set = set(al["honble_members_of_parliaments"].unique())
    ws_set = set(ws["honble_members_of_parliament"].unique())
    wc_set = set(wc["honble_members_of_parliament"].unique())

    ws_matches = ws_set.intersection(al_set)
    wc_matches = wc_set.intersection(al_set)

    sanctioned_match_rate = len(ws_matches) / len(ws_set) if ws_set else 0.0
    completed_match_rate = len(wc_matches) / len(wc_set) if wc_set else 0.0

    logging.info("--- Overlap Summary ---")
    logging.info(f"Sanctioned: {len(ws_matches)} names match Allocated out of {len(ws_set)} total.")
    logging.info(f"Sanctioned Match Rate: {sanctioned_match_rate:.2%}")

    logging.info(f"Completed: {len(wc_matches)} names match Allocated out of {len(wc_set)} total.")
    logging.info(f"Completed Match Rate: {completed_match_rate:.2%}")

    if sanctioned_match_rate < 0.90:
        logging.warning(f"Low match rate in Sanctioned! Sample of unmatched names: {list(ws_set - al_set)[:15]}")

    if completed_match_rate < 0.90:
        logging.warning(f"Low match rate in Completed! Sample of unmatched names: {list(wc_set - al_set)[:15]}")

    diagnose_dataframe(al, "Allocated Limit (cleaned)")
    diagnose_dataframe(ws, "Works Sanctioned (cleaned)")
    diagnose_dataframe(wc, "Works Completed (cleaned)")

    test_ws = grouping_by_mps(ws, "honble_members_of_parliament", "sanction_amount")
    test_wc = grouping_by_mps(wc, "honble_members_of_parliament", "amount_disbursed")

    test_wc.rename(columns={"total_sanction_amount": "total_amount_disbursed"}, inplace=True)
    al.rename(columns={"honble_members_of_parliaments": "honble_members_of_parliament"}, inplace=True)

    step_1 = pd.merge(al, test_ws, on="honble_members_of_parliament", how="left")
    final_merged = pd.merge(step_1, test_wc, on="honble_members_of_parliament", how="left")

    final_merged = final_merged.rename(columns={
    "sanction_work_count_x": "sanctioned_work_count",
    "sanction_work_count_y": "completed_work_count",
    "total_amount_disbursed": "total_disbursed_amount"
    })

    # an MP who shows completed, disbursed works in the Completed table, but has zero matching sanctioned works in the Sanctioned table.

    cols_to_fill = ["total_sanction_amount", "sanctioned_work_count", "total_disbursed_amount", "completed_work_count"]
    final_merged[cols_to_fill] = final_merged[cols_to_fill].fillna(0)

    logging.info(f"Final merged shape: {final_merged.shape}")

    zero_activity = final_merged[(final_merged["sanctioned_work_count"] == 0) & (final_merged["completed_work_count"] == 0)]
    logging.info(f"MPs with zero sanctioned and zero completed works: {len(zero_activity)}")

    orphans = final_merged[(final_merged["sanctioned_work_count"] == 0) & (final_merged["completed_work_count"] > 0)]
    logging.info(f"MPs with completed works but zero matching sanctions: {len(orphans)}")

    df = final_merged.copy()

    df["utilization_rate"] = df["total_disbursed_amount"] / df["allocated_amount"]
    df["sanctioned_backlog"] = df["total_sanction_amount"] - df["total_disbursed_amount"]
    df["completion_ratio"] = np.where(df["sanctioned_work_count"] == 0, np.nan, df["completed_work_count"] / df["sanctioned_work_count"])
    df["has_no_activity"] = (df["sanctioned_work_count"] == 0) & (df["completed_work_count"] == 0)
    df["has_orphan_completions"] = (df["sanctioned_work_count"] == 0) & (df["completed_work_count"] > 0)

    logging.info(f"Zero-activity MPs: {df['has_no_activity'].sum()}")
    logging.info(f"Orphan-completion MPs: {df['has_orphan_completions'].sum()}")


