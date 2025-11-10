import os
import time
import argparse
import sys
import subprocess
import tempfile
import pyttsx3
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import (
    PROCESSED_FILE,
    OUTPUT_DIR,
    SPEECH_RATE_A,
    SPEECH_RATE_B,
    MALE_INDEX,
    TEST_PROCESSED_FILE
)

# ===================== COMMON HELPERS =====================
def ensure_output(path):
    os.makedirs(path, exist_ok=True)


def synthesize_single_utterance(text_file, out_wav, rate, voice_idx):
    """Worker function for subprocess speech generation."""
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"[synth-worker] Failed to read text file: {e}")
        sys.exit(1)

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', int(rate))
        voices = engine.getProperty('voices')
        if 0 <= int(voice_idx) < len(voices):
            engine.setProperty('voice', voices[int(voice_idx)].id)
        engine.save_to_file(text, out_wav)
        engine.runAndWait()
        sys.exit(0)
    except Exception as e:
        print(f"[synth-worker] synth error: {e}")
        sys.exit(1)


# ===================== TRAINING DATA SECTION =====================
def generate_training_speech(timeout_seconds=30, limit=0):
    """Sequentially synthesize speech for training data."""
    training_dir = os.path.join(OUTPUT_DIR, "Training")
    ensure_output(training_dir)
    df = pd.read_csv(PROCESSED_FILE)
    if limit and limit > 0:
        df = df.head(limit)
    if df.empty:
        print("[WEEK2] Empty processed training CSV.")
        return

    synth_script = os.path.abspath(__file__)
    log_file_path = os.path.join(training_dir, "week_redial_train_tts_log.txt")

    # Header for log file
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"TRAINING TTS Log — {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

    writes = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Training Voices"):
        text = str(row.get("clean_text", "")).strip()
        try:
            sender = int(row.get("sender", 0))
        except Exception:
            sender = 0

        if not text:
            continue

        voice_label = "Male_A" if sender == 0 else "Male_B"
        rate = SPEECH_RATE_A if sender == 0 else SPEECH_RATE_B
        filename = f"train_dialogue_{i+1}_{voice_label}.wav"
        path = os.path.join(training_dir, "AudioFiles", filename)
        ensure_output(os.path.dirname(path))

        # Skip if already exists
        if os.path.exists(path):
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{i+1}] {voice_label} | (skipped, exists) {text}\n")
            writes += 1
            continue

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as tf:
            tf.write(text)
            tmp_text_path = tf.name

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

        try:
            os.remove(tmp_text_path)
        except Exception:
            pass

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{i+1}] {voice_label} | {status} | {text}\n")
        if status == "written":
            writes += 1

    print(f"[WEEK2] Training audio generation complete → {training_dir} (files written/checked: {writes})")


# ===================== TESTING DATA SECTION =====================
def generate_testing_speech(timeout_seconds=30, limit=0):
    """Sequentially synthesize speech for testing data."""
    test_output_dir = os.path.join(OUTPUT_DIR, "Testing")
    ensure_output(test_output_dir)

    df_test = pd.read_csv(TEST_PROCESSED_FILE)
    if limit and limit > 0:
        df_test = df_test.head(limit)
    if df_test.empty:
        print("[WEEK2] Empty processed test CSV.")
        return

    synth_script = os.path.abspath(__file__)
    test_log_file = os.path.join(test_output_dir, "week_redial_test_tts_log.txt")

    # Header for log file
    with open(test_log_file, "w", encoding="utf-8") as f:
        f.write(f"TESTING TTS Log — {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

    writes = 0
    for i, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Generating Testing Voices"):
        text = str(row.get("clean_text", "")).strip()
        try:
            sender = int(row.get("sender", 0))
        except Exception:
            sender = 0

        if not text:
            continue

        voice_label = "Male_A" if sender == 0 else "Male_B"
        rate = SPEECH_RATE_A if sender == 0 else SPEECH_RATE_B
        filename = f"test_dialogue_{i+1}_{voice_label}.wav"
        path = os.path.join(test_output_dir, "AudioFiles", filename)
        ensure_output(os.path.dirname(path))

        if os.path.exists(path):
            with open(test_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{i+1}] {voice_label} | (skipped, exists) {text}\n")
            writes += 1
            continue

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as tf:
            tf.write(text)
            tmp_text_path = tf.name

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

        try:
            os.remove(tmp_text_path)
        except Exception:
            pass

        with open(test_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{i+1}] {voice_label} | {status} | {text}\n")
        if status == "written":
            writes += 1

    print(f"[WEEK2] Testing audio generation complete → {test_output_dir} (files written/checked: {writes})")


# ===================== MAIN ENTRY =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-worker", action="store_true", help="Run in synth-worker mode")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per utterance")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both", help="Run mode: train/test/both")
    parser.add_argument('worker_args', nargs='*')
    args = parser.parse_args()

    # synth-worker mode
    if args.synth_worker:
        wa = args.worker_args
        if len(wa) < 4:
            print("synth-worker requires: <text_file> <out_wav> <rate> <voice_index>")
            sys.exit(2)
        synthesize_single_utterance(wa[0], wa[1], wa[2], wa[3])

    # main modes
    if args.mode in ["train", "both"]:
        generate_training_speech(timeout_seconds=args.timeout, limit=args.limit)

    if args.mode in ["test", "both"]:
        generate_testing_speech(timeout_seconds=args.timeout, limit=args.limit)


if __name__ == "__main__":
    main()
