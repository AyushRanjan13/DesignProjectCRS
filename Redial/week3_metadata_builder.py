"""
WEEK 3 — Metadata Creation (REDIAL Dataset)
Goal:
- Build metadata linking text, speaker role, and correct audio paths
- Generate separate metadata for TRAIN and TEST splits
- Maintain the INSPIRED-style folder structure
"""

import os
import pandas as pd
from config import (
    PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    OUTPUT_DIR,
)


def ensure_output():
    """Ensure output directories exist."""
    os.makedirs(os.path.join(OUTPUT_DIR, "Training", "AudioFiles"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "Testing", "AudioFiles"), exist_ok=True)


def build_metadata(split_name: str, csv_file: str, subfolder: str):
    """Build metadata CSV for the given dataset split."""
    ensure_output()
    split_dir = os.path.join(OUTPUT_DIR, subfolder)
    audio_dir = os.path.join(split_dir, "AudioFiles")

    if not os.path.exists(csv_file):
        print(f"[WEEK3-{split_name}] Missing processed CSV → {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"[WEEK3-{split_name}] Empty processed CSV.")
        return

    records = []
    for i, row in df.iterrows():
        sender = int(row["sender"])
        voice_label = "Male_A" if sender == 0 else "Male_B"

        if split_name.lower() == "train":
            audio_file = f"train_dialogue_{i+1}_{voice_label}.wav"
        else:
            audio_file = f"test_dialogue_{i+1}_{voice_label}.wav"

        audio_path = os.path.join(audio_dir, audio_file)

        records.append({
            "dialogue_id": i + 1,
            "audio_path": audio_path,
            "speaker_role": "Recommender" if sender == 0 else "Seeker",
            "voice_label": voice_label,
            "text": row["clean_text"]
        })

    metadata_path = os.path.join(split_dir, f"metadata_{split_name.lower()}.csv")
    pd.DataFrame(records).to_csv(metadata_path, index=False, encoding="utf-8")
    print(f"[WEEK3-{split_name}] Metadata saved → {metadata_path}")


def main():
    print("========= WEEK 3: REDIAL METADATA CREATION =========")
    build_metadata("Train", PROCESSED_FILE, "Training")
    build_metadata("Test", TEST_PROCESSED_FILE, "Testing")
    print("====================================================")


if __name__ == "__main__":
    main()
