"pip install pandas numpy tqdm pyttsx3 librosa scikit-learn joblib torch torchvision torchaudio transformers soundfile sentence-transformers"

import os

# === PATH CONFIG ===
INPUT_TSV = r"C:\Users\Rajeev Ranjan\OneDrive\Desktop\Inspired-master\Inspired-master\data\dialog_data\train.tsv"
PROCESSED_FILE = r"C:\Users\Rajeev Ranjan\Downloads\processed_dialogs_inspired.csv"
OUTPUT_DIR = "week_inspired_output"
TEST_RAW_FILE = r"C:\Users\Rajeev Ranjan\OneDrive\Desktop\Inspired-master\Inspired-master\data\dialog_data\test.tsv"
TEST_PROCESSED_FILE = r"C:\Users\Rajeev Ranjan\Downloads\inspired_test_processed.csv"

# === TTS SETTINGS ===
SPEECH_RATE_A = 170
SPEECH_RATE_B = 190
VOLUME_LEVEL = 1.0
MALE_INDEX = 0

# === OpenSMILE SETTINGS ===
SMIL_EXTRACT_PATH = r"C:\Users\Rajeev Ranjan\OneDrive\Desktop\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"
SMIL_CONFIG = r"C:\Users\Rajeev Ranjan\OneDrive\Desktop\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\config\gemaps\v01a\GeMAPSv01a.conf"

# === OUTPUT FILES (Week 4) ===
ACOUSTIC_FEATURES_CSV = os.path.join(OUTPUT_DIR, "inspired_acoustic_features_opensmile.csv")
PROSODY_FEATURES_CSV = os.path.join(OUTPUT_DIR, "inspired_prosody_features_librosa.csv")
COMBINED_FEATURES_CSV = os.path.join(OUTPUT_DIR, "inspired_combined_acoustic_features.csv")
ENRICHED_METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata_with_acoustic_emotion_inspired.csv")

# === MODEL SETTINGS ===
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "inspired_emotion_model.joblib")

PRETRAINED_MODEL_PATH = None  # set path if you already have one

# === TEXT EMBEDDING SETTINGS (Optional) ===
TEXT_EMB_MODEL = "distilroberta-base"
TEXT_EMB_BATCH = 64
