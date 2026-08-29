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


def mp_name_clean(raw_name: str) -> str:
    cleaned_name = re.sub(r"^(Shri|Smt|Dr\.|Er\.)\s*", "", raw_name).upper()
    return cleaned_name

# al2 = al.copy()
# al2["honble_members_of_parliaments"] = al2["honble_members_of_parliaments"].apply(normalize_mp_name)

# ============================================
# TASK 1 OBSERVATIONS
# ============================================
# All three tables: amount columns loaded as object, need numeric conversion.
# Allocated: allocated_amount has 1 null. No dates, no duplicates.
# Sanctioned: work_description 38 nulls; recommended_date/sanction_date need datetime parsing.
# Completed: work_description 55 nulls, image 5054 nulls (placeholder "Images", low value —
#   dropped), amount_disbursed 6 nulls; completion_date needs datetime parsing.
# sr_no dropped from all tables — redundant sequential counter, no analytical value.
# No "N/A" string-as-null issue found in any table. 0 duplicate rows across all three.

# ============================================
# TASK 2 OBSERVATIONS
# ============================================
# Numeric conversion (coerce): allocated_amount 1→2 nulls, amount_disbursed 6→7 nulls —
#   one additional unparseable value coerced to NaN in each, as expected.
# Dates parsed successfully; 1 unparseable entry per date column turned into NaT.
# 0 exact duplicates found post-cleaning (unchanged from Task 1).
# sr_no and image dropped as planned.

# ============================================
# TASK 3 OBSERVATIONS
# ============================================
# MP names normalized: honorifics stripped, uppercased, whitespace collapsed.
# Match rate vs Allocated: Sanctioned 100%, Completed 100% — all MP names
#   resolved cleanly across tables, no fuzzy-matching fallback needed.
# Caveat: high match rate assumes no residual case/spelling variants beyond what
#   was manually spot-checked — flagged here for the final conclusion (Task 7).