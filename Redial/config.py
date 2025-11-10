"pip install pandas numpy tqdm pyttsx3 librosa scikit-learn joblib torch torchvision torchaudio transformers soundfile sentence-transformers"


import os

# File paths
RAW_FILE = r"C:\Users\Rajeev Ranjan\Downloads\processed_dialogs_updated.csv"
PROCESSED_FILE = r"C:\Users\Rajeev Ranjan\Downloads\processed_dialogs_final.csv"
OUTPUT_DIR = "week_redial_output"
TEST_RAW_FILE = r"C:\Users\Rajeev Ranjan\Downloads\processed_dialogs_test.csv"
TEST_PROCESSED_FILE = r"C:\Users\Rajeev Ranjan\Downloads\redial_test_processed.csv"

# TTS settings
SPEECH_RATE_A = 170
SPEECH_RATE_B = 190
VOLUME_LEVEL = 1.0
MALE_INDEX = 0

# OpenSMILE configuration
SMIL_EXTRACT_PATH = r"C:\Users\Rajeev Ranjan\OneDrive\Desktop\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\bin\SMILExtract.exe"
SMIL_CONFIG = r"C:\Users\Rajeev Ranjan\OneDrive\Desktop\opensmile-3.0.2-windows-x86_64\opensmile-3.0.2-windows-x86_64\config\gemaps\v01a\GeMAPSv01a.conf"

# Output files for Week 4
ACOUSTIC_FEATURES_CSV = os.path.join(OUTPUT_DIR, "redial_acoustic_features_opensmile.csv")
PROSODY_FEATURES_CSV = os.path.join(OUTPUT_DIR, "redial_prosody_features_librosa.csv")
COMBINED_FEATURES_CSV = os.path.join(OUTPUT_DIR, "redial_combined_acoustic_features.csv")
ENRICHED_METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata_with_acoustic_emotion.csv")

# Model configuration
PRETRAINED_MODEL_PATH = None  # keep None to auto-train
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "emotion_model_redial.joblib")
