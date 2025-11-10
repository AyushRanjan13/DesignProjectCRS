import os
import re
import pandas as pd
from config import RAW_FILE, PROCESSED_FILE, TEST_RAW_FILE, TEST_PROCESSED_FILE

# -------------------- CLEANING FUNCTION --------------------
def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,!?'\s]", "", text)
    return text


# -------------------- TRAINING DATA PREPARATION --------------------
def prepare_training_data():
    if not os.path.exists(RAW_FILE):
        print(f"[WEEK1] Raw training file not found → {RAW_FILE}")
        return pd.DataFrame(columns=["sender", "clean_text"])

    df = pd.read_csv(RAW_FILE)
    text_col = "text" if "text" in df.columns else "dialogue"
    if text_col not in df.columns:
        print("[WEEK1] No valid text column found in training data.")
        return pd.DataFrame(columns=["sender", "clean_text"])

    df["clean_text"] = df[text_col].apply(clean_text)

    if "sender" not in df.columns:
        df["sender"] = [i % 2 for i in range(len(df))]

    os.makedirs(os.path.dirname(PROCESSED_FILE) or ".", exist_ok=True)
    df[["sender", "clean_text"]].to_csv(PROCESSED_FILE, index=False)
    print(f"[WEEK1] Training processed file saved → {PROCESSED_FILE}")

    return df


# -------------------- TESTING DATA PREPARATION --------------------
def prepare_testing_data():
    if not os.path.exists(TEST_RAW_FILE):
        print(f"[WEEK1] Test raw file not found → {TEST_RAW_FILE}")
        return pd.DataFrame(columns=["sender", "clean_text"])

    df_test = pd.read_csv(TEST_RAW_FILE)

    # Ensure expected columns
    if "text" not in df_test.columns:
        print("[WEEK1] No valid text column found in test data.")
        return pd.DataFrame(columns=["sender", "clean_text"])

    df_test["clean_text"] = df_test["text"].apply(clean_text)

    if "sender" not in df_test.columns:
        df_test["sender"] = [i % 2 for i in range(len(df_test))]

    # Remove empty cleaned text rows
    df_test = df_test[df_test["clean_text"].str.strip().ne("")]

    os.makedirs(os.path.dirname(TEST_PROCESSED_FILE) or ".", exist_ok=True)
    df_test[["sender", "clean_text", "conversationId"]].to_csv(TEST_PROCESSED_FILE, index=False)
    print(f"[WEEK1] Testing processed file saved → {TEST_PROCESSED_FILE}")

    return df_test


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("========= WEEK 1: REDIAL DATA PREPROCESSING =========")
    train_df = prepare_training_data()
    test_df = prepare_testing_data()
    
