"""
 Extract acoustic (OpenSMILE) + prosody (librosa) features for TRAINING (saved CSVs)
 Train RandomForest model using 100% training data
 Test on full TEST set — NO CSVs are saved for testing
 Console output (all print logs + evaluation) saved to redial_testing_report.txt
"""

import os
import subprocess
import shutil
import sys
from io import StringIO
from datetime import datetime

import pandas as pd
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from config import (
    SMIL_EXTRACT_PATH, SMIL_CONFIG, OUTPUT_DIR,
    ACOUSTIC_FEATURES_CSV, PROSODY_FEATURES_CSV,
    COMBINED_FEATURES_CSV, ENRICHED_METADATA_CSV,
    MODEL_SAVE_PATH
)

TRAIN_METADATA_FILE = os.path.join(OUTPUT_DIR, "Training", "metadata_train.csv")
TEST_METADATA_FILE = os.path.join(OUTPUT_DIR, "Testing", "metadata_test.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "redial_testing_report.txt")

# -------------------- Utility --------------------
def find_smil_executable():
    for name in ("SMILExtract.exe", "SMILExtract"):
        path = shutil.which(name)
        if path:
            return path
    if SMIL_EXTRACT_PATH and os.path.exists(SMIL_EXTRACT_PATH):
        return SMIL_EXTRACT_PATH
    return None


def run_opensmile_extract(wav, out_csv, smil_exec, smil_conf):
    cmd = [smil_exec, "-C", smil_conf, "-I", wav, "-csvoutput", out_csv]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def resolve_audio_paths(meta_df, metadata_file_path):
    meta_dir = os.path.dirname(os.path.abspath(metadata_file_path))
    output_dir_abs = os.path.abspath(OUTPUT_DIR)
    repo_base = os.path.abspath(os.path.join(output_dir_abs, os.pardir))

    resolved = []
    for p in meta_df["audio_path"].astype(str).tolist():
        p_norm = os.path.normpath(p)
        if os.path.isabs(p_norm):
            resolved.append(os.path.abspath(p_norm))
        elif p_norm.startswith(os.path.basename(OUTPUT_DIR)):
            resolved.append(os.path.abspath(os.path.join(repo_base, p_norm)))
        else:
            resolved.append(os.path.abspath(os.path.join(meta_dir, p_norm)))
    return resolved


# -------------------- Feature Extraction --------------------
def extract_prosody_librosa(wav):
    try:
        y, sr = librosa.load(wav, sr=None)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        tempo = librosa.beat.tempo(y=y, sr=sr)
        return {
            "audio_path": os.path.abspath(wav),
            "f0_mean": float(np.mean(f0_clean)) if f0_clean.size else 0.0,
            "f0_std": float(np.std(f0_clean)) if f0_clean.size else 0.0,
            "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
            "spec_cent_mean": float(np.mean(spec_cent)) if spec_cent.size else 0.0,
            "tempo": float(tempo[0]) if len(tempo) else 0.0
        }
    except Exception:
        return {"audio_path": os.path.abspath(wav),
                "f0_mean": 0.0, "f0_std": 0.0, "rms_mean": 0.0,
                "spec_cent_mean": 0.0, "tempo": 0.0}


def compute_all_prosody(audio_paths, out_csv):
    feats = [extract_prosody_librosa(p) for p in tqdm(audio_paths, desc="Librosa prosody")]
    df = pd.DataFrame(feats)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def batch_opensmile_extract(audio_files, out_csv_all, smil_exec, smil_conf):
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


# -------------------- Emotion Modeling --------------------
def rule_based_label(row):
    if row.get("rms_mean", 0) > 0.03 and row.get("tempo", 0) > 130:
        return "high_arousal"
    if row.get("f0_mean", 0) > 220:
        return "high_pitch"
    return "neutral"


def train_emotion_model(df_train, model_out_path):
    X_train = df_train.select_dtypes(include=[np.number]).fillna(0)
    y_train = df_train.apply(rule_based_label, axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y_train)

    joblib.dump((clf, scaler), model_out_path)
    print(f"[WEEK4] Trained model saved → {model_out_path}")
    return clf, scaler


# -------------------- Evaluation --------------------
def evaluate_on_test(df_test, model_path):
    clf, scaler = joblib.load(model_path)
    X_test = df_test.select_dtypes(include=[np.number]).fillna(0)
    y_true = df_test.apply(rule_based_label, axis=1)
    y_pred = clf.predict(scaler.transform(X_test))

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    print("\n================= [WEEK4 TEST PERFORMANCE] =================")
    print(f"Samples: {len(df_test)}")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("============================================================\n")
    return acc, report


# -------------------- Main Pipeline --------------------
def week4_pipeline():
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    print("========= WEEK 4: REDIAL ACOUSTIC + EMOTION MODELING =========")

    if not os.path.exists(TRAIN_METADATA_FILE):
        print(f"[WEEK4] Missing training metadata: {TRAIN_METADATA_FILE}")
        sys.stdout = old_stdout
        return

    meta_train = pd.read_csv(TRAIN_METADATA_FILE)
    resolved_train = resolve_audio_paths(meta_train, TRAIN_METADATA_FILE)
    audio_train = [p for p in resolved_train if os.path.exists(p)]
    if not audio_train:
        print("[WEEK4] No accessible training audio files found.")
        sys.stdout = old_stdout
        return

    smil_exec = find_smil_executable()

    print("[WEEK4] Extracting features for training samples...")
    prosody_df = compute_all_prosody(audio_train, PROSODY_FEATURES_CSV)
    opensmile_df = batch_opensmile_extract(audio_train, ACOUSTIC_FEATURES_CSV, smil_exec, SMIL_CONFIG)

    combined = prosody_df.merge(opensmile_df, on="audio_path", how="left")
    combined.to_csv(COMBINED_FEATURES_CSV, index=False)

    clf, scaler = train_emotion_model(combined, MODEL_SAVE_PATH)

    print("[WEEK4] Generating enriched metadata with predicted emotions...")

    numeric_features = combined.select_dtypes(include=[np.number]).fillna(0)
    preds = clf.predict(scaler.transform(numeric_features))
    combined["predicted_emotion"] = preds

    enriched = meta_train.merge(combined, on="audio_path", how="left")
    enriched.to_csv(ENRICHED_METADATA_CSV, index=False)
    print(f"[WEEK4] Enriched metadata saved → {ENRICHED_METADATA_CSV}")

    # ---- Testing (no CSVs saved) ----
    if os.path.exists(TEST_METADATA_FILE):
        meta_test = pd.read_csv(TEST_METADATA_FILE)
        resolved_test = resolve_audio_paths(meta_test, TEST_METADATA_FILE)
        audio_test = [p for p in resolved_test if os.path.exists(p)]
        if audio_test:
            prosody_test = [extract_prosody_librosa(p) for p in tqdm(audio_test, desc="Test Prosody")]
            df_test = pd.DataFrame(prosody_test)
            acc, report = evaluate_on_test(df_test, MODEL_SAVE_PATH)
            print(f"[WEEK4] Test accuracy: {acc:.4f}")

    sys.stdout = old_stdout
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(mystdout.getvalue())
    print(f"\nAll logs and test report saved → {REPORT_FILE}")


if __name__ == "__main__":
    week4_pipeline()
