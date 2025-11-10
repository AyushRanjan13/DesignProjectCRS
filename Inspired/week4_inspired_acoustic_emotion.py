"""
WEEK 4 — Acoustic + Emotion Modeling (INSPIRED Dataset)
Goal:
- Extract acoustic + prosody features (OpenSMILE + Librosa)
- Train emotion model using 100% training data
- Evaluate on separate testing dataset (no feature CSVs saved for testing)
- Save test evaluation report to a .txt file
"""

import os
import subprocess
import shutil
import pandas as pd
import numpy as np
import librosa
import joblib
import tempfile
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from config_inspired import (
    SMIL_EXTRACT_PATH, SMIL_CONFIG, OUTPUT_DIR,
    ACOUSTIC_FEATURES_CSV, PROSODY_FEATURES_CSV,
    COMBINED_FEATURES_CSV, ENRICHED_METADATA_CSV,
    MODEL_SAVE_PATH
)

# ============================ FILE PATHS ============================

TRAIN_METADATA_FILE = os.path.join(OUTPUT_DIR, "Training", "metadata_train.csv")
TEST_METADATA_FILE = os.path.join(OUTPUT_DIR, "Testing", "metadata_test.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "inspired_test_report.txt")


# ============================ OPENSMILE EXTRACTION ============================

def find_smil_executable():
    """Locate OpenSMILE executable."""
    if os.path.exists(SMIL_EXTRACT_PATH):
        return SMIL_EXTRACT_PATH
    return shutil.which("SMILExtract.exe")


def run_opensmile_extract(wav, out_csv, smil_exec, smil_conf):
    """Run OpenSMILE on a single WAV file."""
    cmd = [smil_exec, "-C", smil_conf, "-I", wav, "-csvoutput", out_csv]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def batch_opensmile_extract(audio_files, out_csv_all, smil_exec, smil_conf):
    """Batch extract OpenSMILE acoustic features."""
    if not smil_exec:
        print("[WEEK4] OpenSMILE not found.")
        return pd.DataFrame()
    rows = []
    for idx, wav in enumerate(tqdm(audio_files, desc="OpenSMILE extracting")):
        tmp = f"{out_csv_all}.tmp{idx}.csv"
        if run_opensmile_extract(wav, tmp, smil_exec, smil_conf) and os.path.exists(tmp):
            df_row = pd.read_csv(tmp)
            df_row["audio_path"] = os.path.abspath(wav)
            rows.append(df_row)
            os.remove(tmp)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_csv_all, index=False)
    return combined


# ============================ LIBROSA PROSODY EXTRACTION ============================

def extract_prosody_librosa(wav):
    """Extract prosody features using Librosa."""
    try:
        y, sr = librosa.load(wav, sr=None)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        return {
            "f0_mean": np.mean(f0) if f0.size else 0,
            "f0_std": np.std(f0) if f0.size else 0,
            "rms_mean": np.mean(librosa.feature.rms(y=y)),
            "spec_cent_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
            "audio_path": os.path.abspath(wav)
        }
    except Exception:
        return {"f0_mean": 0, "f0_std": 0, "rms_mean": 0,
                "spec_cent_mean": 0, "tempo": 0,
                "audio_path": os.path.abspath(wav)}


def compute_all_prosody(audio_paths, out_csv):
    """Compute prosody features for all audio files."""
    feats = [extract_prosody_librosa(p) for p in tqdm(audio_paths, desc="Librosa prosody")]
    df = pd.DataFrame(feats)
    df.to_csv(out_csv, index=False)
    return df


# ============================ EMOTION MODELING ============================

def rule_based_label(row):
    """Assign simple rule-based emotion labels."""
    if row["rms_mean"] > 0.03 and row["tempo"] > 130:
        return "high_arousal"
    if row["f0_mean"] > 220:
        return "high_pitch"
    return "neutral"


def train_emotion_model(df_train, model_out_path):
    """Train Random Forest on full training data."""
    X_train = df_train.select_dtypes(include=[np.number]).fillna(0)
    y_train = df_train.apply(rule_based_label, axis=1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    # Save model and scaler
    joblib.dump((clf, scaler), model_out_path)
    print(f"[WEEK4] Trained model → {model_out_path}")
    return clf, scaler


def evaluate_on_test(df_test, model_path):
    """Evaluate trained model on testing data and save report."""
    clf, scaler = joblib.load(model_path)
    X_test = df_test.select_dtypes(include=[np.number]).fillna(0)
    y_true = df_test.apply(rule_based_label, axis=1)
    y_pred = clf.predict(scaler.transform(X_test))

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    # ---- Print to console ----
    print("\n================= [WEEK4 TEST PERFORMANCE] =================")
    print(f"Samples: {len(df_test)}")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("============================================================\n")

    # ---- Save to report file ----
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("================= [WEEK4 TEST PERFORMANCE] =================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {len(df_test)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\n============================================================\n")

    print(f"[WEEK4] Test report saved → {REPORT_FILE}")
    return y_true, y_pred, acc, report


# ============================ MAIN PIPELINE ============================

def week4_pipeline():
    """Full Week 4 acoustic + emotion modeling pipeline."""
    # ==== TRAINING DATASET ====
    if not os.path.exists(TRAIN_METADATA_FILE):
        print("[WEEK4] Training metadata missing.")
        return

    meta_train = pd.read_csv(TRAIN_METADATA_FILE)
    audio_train = [p for p in meta_train["audio_path"] if os.path.exists(p)]
    smil_exec = find_smil_executable()

    # 1. Extract training features
    opensmile_df = batch_opensmile_extract(audio_train, ACOUSTIC_FEATURES_CSV, smil_exec, SMIL_CONFIG)
    prosody_df = compute_all_prosody(audio_train, PROSODY_FEATURES_CSV)

    # 2. Combine and save
    combined_train = prosody_df.merge(opensmile_df, on="audio_path", how="left")
    combined_train.to_csv(COMBINED_FEATURES_CSV, index=False)

    # 3. Train model
    clf, scaler = train_emotion_model(combined_train, MODEL_SAVE_PATH)

    # 4. Add emotion predictions and save enriched metadata
    combined_train["predicted_emotion"] = combined_train.apply(rule_based_label, axis=1)
    enriched_train = meta_train.merge(combined_train, on="audio_path", how="right")
    enriched_train.to_csv(ENRICHED_METADATA_CSV, index=False)
    print(f"[WEEK4] Enriched training metadata saved → {ENRICHED_METADATA_CSV}")

    # ==== TESTING DATASET ====
    if not os.path.exists(TEST_METADATA_FILE):
        print("[WEEK4] Test metadata missing, skipping test evaluation.")
        return

    meta_test = pd.read_csv(TEST_METADATA_FILE)
    audio_test = [p for p in meta_test["audio_path"] if os.path.exists(p)]

    print("\n[WEEK4] Extracting test features (no CSVs will be saved)...")
    test_prosody = [extract_prosody_librosa(p) for p in tqdm(audio_test, desc="Librosa test prosody")]

    smil_exec = find_smil_executable()
    test_opensmile = []
    for p in tqdm(audio_test, desc="OpenSMILE test features"):
        tmp = os.path.join(tempfile.gettempdir(), "tmp_test.csv")
        if run_opensmile_extract(p, tmp, smil_exec, SMIL_CONFIG) and os.path.exists(tmp):
            df_row = pd.read_csv(tmp)
            df_row["audio_path"] = os.path.abspath(p)
            test_opensmile.append(df_row)
            os.remove(tmp)

    if test_opensmile:
        df_opensmile_test = pd.concat(test_opensmile, ignore_index=True)
        df_prosody_test = pd.DataFrame(test_prosody)
        df_test_combined = df_prosody_test.merge(df_opensmile_test, on="audio_path", how="left")
    else:
        df_test_combined = pd.DataFrame(test_prosody)

    # Evaluate and save report
    y_true, y_pred, acc, report = evaluate_on_test(df_test_combined, MODEL_SAVE_PATH)
    print(f"[WEEK4] Test evaluation complete — Accuracy: {acc:.4f}")


# ============================ ENTRY POINT ============================

if __name__ == "__main__":
    week4_pipeline()
