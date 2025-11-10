"""
WEEK 3 — Metadata Creation (INSPIRED Dataset)
Goal:
- Build metadata linking text, speaker role, and audio paths
- Generate separate metadata for TRAIN and TEST splits
"""

import os
import pandas as pd
from config_inspired import (
    PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    OUTPUT_DIR,
)

def ensure_output():
    """Ensure output directories exist."""
    os.makedirs(os.path.join(OUTPUT_DIR, "Training", "AudioFiles"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "Testing", "AudioFiles"), exist_ok=True)


def build_metadata(split_name: str, csv_file: str, subfolder: str):
    """Build metadata CSV for a given dataset split."""
    split_dir = os.path.join(OUTPUT_DIR, subfolder)
    audio_dir = os.path.join(split_dir, "AudioFiles")
    ensure_output()

    if not os.path.exists(csv_file):
        print(f"[WEEK3-{split_name}] Missing processed CSV → {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"[WEEK3-{split_name}] Empty processed CSV.")
        return

    records = []
    for i, row in df.iterrows():
        speaker = str(row.get("speaker", "")).upper().strip()
        # Voice assignment based on speaker role
        v_label = "Male_A" if "RECOMMENDER" in speaker else "Male_B"

        # Audio file naming
        audio_file = f"inspired_{split_name.lower()}_dialogue_{i+1}_{v_label}.wav"
        audio_path = os.path.join(audio_dir, audio_file)

        records.append({
            "dialogue_id": i + 1,
            "audio_path": audio_path,
            "speaker_role": speaker.title(),
            "voice_label": v_label,
            "text": row.get("clean_text", "")
        })

    metadata_path = os.path.join(split_dir, f"metadata_{split_name.lower()}.csv")
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    print(f"[WEEK3-{split_name}] Metadata saved → {metadata_path}")


def main():
    print("=== WEEK 3: Generating Metadata for INSPIRED Dataset ===")
    build_metadata("Train", PROCESSED_FILE, "Training")
    build_metadata("Test", TEST_PROCESSED_FILE, "Testing")
    print("=== Metadata Creation Completed Successfully ===")


if __name__ == "__main__":
    main()
