"""
WEEK 2 — Text-to-Speech Generation (INSPIRED Dataset)
Goal:
- Convert cleaned dialogues from both TRAIN and TEST datasets into audio (.wav)
- Use dual male voices (Male_A and Male_B)
- Save files in structured folders (Training/ and Testing/ each having AudioFiles/)
- Maintain separate log files for each split
"""

import os
import time
import argparse
import sys
import subprocess
import tempfile
import pandas as pd
from tqdm import tqdm
import pyttsx3
from datetime import datetime
from config_inspired import (
    PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    OUTPUT_DIR,
    SPEECH_RATE_A,
    SPEECH_RATE_B,
    VOLUME_LEVEL,
    MALE_INDEX
)

# === Helper: Ensure subdirectories exist ===
def ensure_output():
    os.makedirs(os.path.join(OUTPUT_DIR, "Training", "AudioFiles"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "Testing", "AudioFiles"), exist_ok=True)


# === Generic TTS Generator ===
def generate_tts(split_name: str, input_file: str, subfolder: str, timeout_seconds=30, limit=0):
    """Generate TTS for a dataset split (train or test)."""

    ensure_output()

    if not os.path.exists(input_file):
        print(f"[WEEK2-{split_name}] File not found → {input_file}")
        return

    df = pd.read_csv(input_file)
    if limit and limit > 0:
        df = df.head(limit)

    if df.empty:
        print(f"[WEEK2-{split_name}] Processed CSV is empty.")
        return

    split_dir = os.path.join(OUTPUT_DIR, subfolder)
    audio_dir = os.path.join(split_dir, "AudioFiles")
    os.makedirs(audio_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(split_dir, f"week_inspired_{split_name.lower()}_tts_log.txt")
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"TTS Log — {split_name.upper()} DATASET ({datetime.now()})\n{'='*60}\n\n")

    synth_script = os.path.abspath(__file__)
    written = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {split_name} Audio"):
        text = str(row.get("clean_text", "")).strip()
        speaker = str(row.get("speaker", "")).upper()

        if not text:
            continue

        v_label = "Male_A" if "RECOMMENDER" in speaker else "Male_B"
        rate = SPEECH_RATE_A if v_label == "Male_A" else SPEECH_RATE_B
        filename = f"inspired_{split_name.lower()}_dialogue_{i+1}_{v_label}.wav"
        path = os.path.join(audio_dir, filename)

        # Skip existing
        if os.path.exists(path):
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"[{i+1}] {v_label} ({speaker}) | skipped (exists)\n")
            written += 1
            continue

        # Temporary text file
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as tf:
            tf.write(text)
            tmp_text_path = tf.name

        # Run synth-worker
        cmd = [sys.executable, synth_script, "--synth-worker", tmp_text_path, path, str(rate), str(MALE_INDEX)]
        try:
            subprocess.run(cmd, check=True, timeout=timeout_seconds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            status = "written" if os.path.exists(path) else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except subprocess.CalledProcessError as e:
            status = f"error:{e.returncode}"
        except Exception as e:
            status = f"error:{e}"

        # Cleanup temp
        try:
            os.remove(tmp_text_path)
        except Exception:
            pass

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"[{i+1}] {v_label} ({speaker}) | {status}\n{text}\n\n")

        if status == "written":
            written += 1

    print(f"[WEEK2-{split_name}] Audio generation complete → {audio_dir} (files processed: {written})")


# === Synth-worker mode ===
def synth_worker_mode(args):
    """Internal subprocess: performs one text-to-speech generation safely."""
    wa = args.worker_args
    if len(wa) < 4:
        print("synth-worker requires: <text_file> <out_wav> <rate> <voice_index>")
        sys.exit(2)

    text_file, out_wav, rate_s, voice_idx_s = wa
    try:
        rate = int(rate_s)
        voice_idx = int(voice_idx_s)
    except Exception:
        rate = SPEECH_RATE_A
        voice_idx = MALE_INDEX

    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"[synth-worker] Failed to read text file: {e}")
        sys.exit(1)

    try:
        engine = pyttsx3.init()
        try:
            engine.setProperty('rate', rate)
            voices = engine.getProperty('voices')
            if 0 <= voice_idx < len(voices):
                engine.setProperty('voice', voices[voice_idx].id)
            engine.setProperty('volume', VOLUME_LEVEL)
        except Exception:
            pass
        engine.save_to_file(text, out_wav)
        engine.runAndWait()
        sys.exit(0)
    except Exception as e:
        print(f"[synth-worker] synth error: {e}")
        sys.exit(1)


# === CLI Entrypoint ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-worker", action="store_true",
                        help="Run in synth-worker mode: expects <text_file> <out_wav> <rate> <voice_index>")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N rows for testing (0 = all)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout (s) per utterance subprocess")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both",
                        help="Select which dataset to generate TTS for")
    parser.add_argument('worker_args', nargs='*')
    args = parser.parse_args()

    if args.synth_worker:
        synth_worker_mode(args)
    else:
        if args.split in ("train", "both"):
            generate_tts("Train", PROCESSED_FILE, "Training", args.timeout, args.limit)
        if args.split in ("test", "both"):
            generate_tts("Test", TEST_PROCESSED_FILE, "Testing", args.timeout, args.limit)


if __name__ == "__main__":
    main()
