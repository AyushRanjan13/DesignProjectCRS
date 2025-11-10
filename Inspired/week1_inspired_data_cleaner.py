"""
WEEK 1 — Data Cleaning (INSPIRED Dataset)

Goals:
- Clean both training and testing dialogues.
- Remove emojis, IDs, and unwanted noise.
- Save as processed CSVs for both datasets.
"""

import pandas as pd
import re
import os
from config_inspired import (
    INPUT_TSV,
    PROCESSED_FILE,
    TEST_RAW_FILE,
    TEST_PROCESSED_FILE
)

def clean_text(text):
    """Remove unwanted symbols, emojis, and noise from text."""
    if pd.isna(text):
        return ""
    text = re.sub(r'@\d+', '', text)                # remove IDs like @123
    text = re.sub(r'[^\w\s.,?!\'"-]', '', text)     # remove special chars
    text = re.sub(r'\s+', ' ', text)                # normalize spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)       # remove non-ASCII
    return text.strip()


# ------------------ TRAINING DATA ------------------ #
def prepare_inspired_train_data():
    """Load train.tsv, clean it, and save as processed CSV."""
    if not os.path.exists(INPUT_TSV):
        raise FileNotFoundError(f"[WEEK1] Train file not found: {INPUT_TSV}")

    df_train = pd.read_csv(INPUT_TSV, sep="\t", quoting=3, encoding="utf-8")

    if "text" not in df_train.columns:
        raise KeyError("Expected 'text' column not found in train.tsv")

    df_train["clean_text"] = df_train["text"].apply(clean_text)
    df_train[["dialog_id", "speaker", "clean_text"]].to_csv(PROCESSED_FILE, index=False)

    print(f"[WEEK1] Train processed file saved → {PROCESSED_FILE}")


# ------------------ TESTING DATA ------------------ #
def prepare_inspired_test_data():
    """Load test.tsv, clean it, and save as processed CSV."""
    if not os.path.exists(TEST_RAW_FILE):
        raise FileNotFoundError(f"[WEEK1] Test file not found: {TEST_RAW_FILE}")

    df_test = pd.read_csv(TEST_RAW_FILE, sep="\t", quoting=3, encoding="utf-8")

    if "text" not in df_test.columns:
        raise KeyError("Expected 'text' column not found in test.tsv")

    df_test["clean_text"] = df_test["text"].apply(clean_text)
    df_test[["dialog_id", "speaker", "clean_text"]].to_csv(TEST_PROCESSED_FILE, index=False)

    print(f"[WEEK1] Test processed file saved → {TEST_PROCESSED_FILE}")


# ------------------ MAIN EXECUTION ------------------ #
if __name__ == "__main__":
    prepare_inspired_train_data()
    prepare_inspired_test_data()
