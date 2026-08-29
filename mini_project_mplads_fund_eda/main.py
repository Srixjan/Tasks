import logging
logging.basicConfig(level=logging.INFO)

from src.data_pipeline import FileMissingValueError, FileValidationError

from src.data_pipeline import diagnose_dataframe, load_results

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

    al["honble_members_of_parliaments"] = al["honble_members_of_parliaments"].apply(mp_name_clean)
    ws["honble_members_of_parliament"] = ws["honble_members_of_parliament"].apply(mp_name_clean)
    wc["honble_members_of_parliament"] = wc["honble_members_of_parliament"].apply(mp_name_clean)

    diagnose_dataframe(al, "Allocated Limit (cleaned)")
    diagnose_dataframe(ws, "Works Sanctioned (cleaned)")
    diagnose_dataframe(wc, "Works Completed (cleaned)")