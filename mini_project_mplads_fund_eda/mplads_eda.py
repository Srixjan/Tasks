import logging
import pandas as pd
import numpy as np
import os
import re

logging.basicConfig(level=logging.INFO)

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
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        logging.info(f"Loaded {filepath} — shape: {df.shape}")
        return df


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    dfCols = list(df.columns)
    columns_cleaned = []

    for col in dfCols:
        cleaned_names = re.sub(r"[\.\'\()\₹]", '', str(col)).lower().strip()
        cleaned_names = cleaned_names.replace(" ", "_")
        columns_cleaned.append(cleaned_names)

    df.columns = columns_cleaned
    return df


def diagnose_dataframe(df: pd.DataFrame, name: str):
    logging.info(f"\n Dataframe -> {name}\n")
    logging.info(f" Shape: \n{df.shape}\n")
    logging.info(f" Dataframe Datatypes \n{df.dtypes}\n")
    logging.info(f" Dataframe null columns \n{df.isna().sum()[df.isna().sum()>0]}\n")
    logging.info(f" Datarframe duplicates \n{df.duplicated().sum()}")


def numeric_conversion(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for cols in columns:
        df[cols] = pd.to_numeric(df[cols], errors="coerce", downcast="float")
    return df


def replace_strings_with_real_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("N/A", np.nan)
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col_name in df.columns:
        if df[col_name].dtype == "object":
            df[col_name] = df[col_name].str.strip()
            df[col_name] = df[col_name].str.replace(r"\t", " ", regex=True)
    return df


def convert_real_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for cols in columns:
        initial_count = df[cols].isna().sum()
        df[cols] = pd.to_datetime(df[cols], errors="coerce", format="%d-%b-%Y")
        failed_count = df[cols].isna().sum()
        failed = failed_count - initial_count
        logging.info(f"Column '{cols}': {failed} unparseable entries turned into NaT.")
    return df


def drop_duplicate_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logging.info(f"{name}: dropped {before - after} duplicate rows")
    return df


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

    # ============================================
    # STEP 5 — OBSERVATIONS (Task 1)
    # ============================================
    # --- Allocated Limit for Honble MPs ---
    # - All columns are dtype object, including allocated_amount (should be numeric)
    # - allocated_amount has 1 null value
    # - sr_no is a redundant sequential counter stored as text — will drop in Task 2
    # - No "N/A" string issue found
    # - No date columns in this table
    # - 0 duplicate rows

    # --- Works Sanctioned ---
    # - All columns are dtype object, including sanction_amount (should be numeric)
    # - work_description has 38 nulls
    # - recommended_date and sanction_date are stored as object, need datetime conversion
    # - sr_no is a redundant sequential counter — will drop in Task 2
    # - Checked work_description for "N/A" strings — none found, looks like genuine free text
    # - Inconsistent casing in work_description (ALL CAPS / Title Case / lowercase mixed) —
    #   not a join key, so leaving as-is, just noting it
    # - 0 duplicate rows

    # --- Works Completed ---
    # - All columns are dtype object, including amount_disbursed (should be numeric)
    # - work_description has 55 nulls, image has 5054 nulls, amount_disbursed has 6 nulls
    # - completion_date is stored as object, needs datetime conversion
    # - sr_no is a redundant sequential counter — will drop in Task 2
    # - image column checked — no "N/A" strings, but values are either real NaN or the
    #   placeholder text "Images" (non-informative) — will drop this column in Task 2
    # - 0 duplicate rows

    # ============================================
    # TASK 2 — PER-TABLE CLEANING
    # ============================================
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

    diagnose_dataframe(al, "Allocated Limit (cleaned)")
    diagnose_dataframe(ws, "Works Sanctioned (cleaned)")
    diagnose_dataframe(wc, "Works Completed (cleaned)")