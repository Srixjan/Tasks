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
        cleaned_names = re.sub(r"[\.\'\()\₹]", '',str(col)).lower().strip()
        cleaned_names = cleaned_names.replace(" ", "_")
        columns_cleaned.append(cleaned_names)

    df.columns = (columns_cleaned)
    return df




def diagnose_dataframe(df: pd.DataFrame, name: str):
    logging.info(f"\n Dataframe -> {name}\n")
    logging.info(f" Shape: \n{df.shape}\n")
    logging.info(f" Dataframe Datatypes \n{df.dtypes}\n")
    logging.info(f" Dataframe null columns \n{df.isna().sum()[df.isna().sum()>0]}\n")
    logging.info(f" Datarframe duplicates \n{df.duplicated().sum()}")

    

if __name__ == "__main__":
    try:
        al = load_results("data/raw/Allocated Limit for Honble MPs.csv")
        wc = load_results("data/raw/Works Completed.csv")
        ws = load_results("data/raw/Works Sanctioned.csv")

    except FileValidationError as e:
        logging.error(f"Loading Failed: {e}")
        raise

clean_dataframe_columns(al).columns
clean_dataframe_columns(wc).columns
clean_dataframe_columns(ws).columns

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