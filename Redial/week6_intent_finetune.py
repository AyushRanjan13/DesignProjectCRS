"""
WEEK 6 — Intent Detection Fine-tuning (ReDial)
Goal:
- Fine-tune joint speech-text embeddings for CRS intent classes
- Use ReDial Week-5 aligned embeddings as input features
- Train and validate an intent classifier (intent-aware speech model)
- Save model + training/validation report
"""

import os
import sys
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

from config import OUTPUT_DIR


# --------- PATHS (ReDial) - CORRECTED TO MATCH WEEK 5 ---------
TRAIN_META = os.path.join(OUTPUT_DIR, "metadata_with_acoustic_emotion.csv")

# This is the EXACT path from Week 5 line 27
TRAIN_EMB_PATH = os.path.join(OUTPUT_DIR, "speech_text_embeddings_train.csv")

# Week 5 encoder model (line 25)
WEEK5_ENCODER_PATH = os.path.join(OUTPUT_DIR, "week5_encoder.joblib")

# Week 6 outputs
INTENT_MODEL_PATH = os.path.join(OUTPUT_DIR, "week6_intent_model.joblib")
INTENT_REPORT_PATH = os.path.join(OUTPUT_DIR, "week6_intent_report.txt")


# --------- SIMPLE RULE-BASED INTENT TAGGER ---------
def rule_based_intent(text: str) -> str:
    """
    Very lightweight intent labeling for ReDial CRS.
    You can later refine/extend the keyword sets or
    replace this with human-annotated labels.

    Returns one of:
    - GREETING
    - ASK_RECOMMENDATION
    - GIVE_PREFERENCE
    - ASK_INFO
    - FEEDBACK_POS
    - FEEDBACK_NEG
    - THANK
    - CLOSING
    - OTHER
    """
    t = (text or "").strip().lower()

    if not t:
        return "OTHER"

    # Greeting
    greeting_kw = ["hi", "hello", "hey", "good morning", "good evening"]
    if any(kw in t for kw in greeting_kw):
        return "GREETING"

    # Thank you
    thanks_kw = ["thank you", "thanks", "thx", "tysm"]
    if any(kw in t for kw in thanks_kw):
        return "THANK"

    # Closing
    closing_kw = ["bye", "goodbye", "see you", "talk later", "have a nice day"]
    if any(kw in t for kw in closing_kw):
        return "CLOSING"

    # Ask for recommendation
    ask_rec_kw = [
        "recommend", "suggest", "what should i watch",
        "any movie", "any film", "can you recommend"
    ]
    if any(kw in t for kw in ask_rec_kw) and "?" in t:
        return "ASK_RECOMMENDATION"

    # Provide preferences
    pref_kw = [
        "i like", "i love", "my favourite", "my favorite",
        "i prefer", "i enjoy", "i usually watch"
    ]
    if any(kw in t for kw in pref_kw):
        return "GIVE_PREFERENCE"

    # Ask information about a movie
    ask_info_kw = [
        "what is it about", "what's it about", "what is this about",
        "who stars", "who is in it", "is it good", "how is it"
    ]
    if any(kw in t for kw in ask_info_kw):
        return "ASK_INFO"

    # Positive feedback
    pos_kw = ["great", "awesome", "amazing", "nice", "good choice", "sounds good"]
    if any(kw in t for kw in pos_kw):
        return "FEEDBACK_POS"

    # Negative feedback
    neg_kw = ["didn't like", "dont like", "don't like", "boring",
              "bad", "not good", "hate"]
    if any(kw in t for kw in neg_kw):
        return "FEEDBACK_NEG"

    return "OTHER"


# --------- CORE PIPELINE ---------
def check_output_directory():
    """Check and list all files in output directory"""
    print(f"[WEEK6] Checking output directory: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"[WEEK6] ERROR: Output directory does not exist: {OUTPUT_DIR}")
        print(f"[WEEK6] Please create it or run previous weeks first.")
        return False
    
    files = os.listdir(OUTPUT_DIR)
    print(f"[WEEK6] Found {len(files)} files in output directory:")
    for f in sorted(files):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"[WEEK6]   - {f} ({size_mb:.2f} MB)")
    print()
    return True


def load_inputs():
    """Load Week 4 metadata and Week 5 embeddings with proper validation"""
    print("[WEEK6] Validating input files...")
    
    # First check if output directory exists
    if not check_output_directory():
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR}")
    
    # Check metadata file
    if not os.path.exists(TRAIN_META):
        raise FileNotFoundError(
            f"\n[WEEK6] ERROR: Week-4 enriched metadata not found!\n"
            f"Expected: {TRAIN_META}\n"
            f"Please run Week 4 pipeline first:\n"
            f"  python week4_acoustic_emotion.py\n"
        )
    else:
        print(f"[WEEK6] ✓ Found metadata: {TRAIN_META}")

    # Check embeddings file
    if not os.path.exists(TRAIN_EMB_PATH):
        raise FileNotFoundError(
            f"\n[WEEK6] ERROR: Week-5 embeddings not found!\n"
            f"Expected: {TRAIN_EMB_PATH}\n"
            f"Please run Week 5 pipeline first:\n"
            f"  python week5_speech_text_representation.py\n"
        )
    else:
        print(f"[WEEK6] ✓ Found embeddings: {TRAIN_EMB_PATH}")
    
    # Check encoder (warning only, not required for training)
    if not os.path.exists(WEEK5_ENCODER_PATH):
        print(f"[WEEK6] ⚠ Warning: Week 5 encoder not found at {WEEK5_ENCODER_PATH}")
        print("[WEEK6] This is needed for inference but not for training.")
    else:
        print(f"[WEEK6] ✓ Found encoder: {WEEK5_ENCODER_PATH}")

    print(f"\n[WEEK6] Loading metadata from: {os.path.basename(TRAIN_META)}")
    meta = pd.read_csv(TRAIN_META)
    
    print(f"[WEEK6] Loading embeddings from: {os.path.basename(TRAIN_EMB_PATH)}")
    emb = pd.read_csv(TRAIN_EMB_PATH)

    print(f"[WEEK6] Metadata rows: {len(meta)}")
    print(f"[WEEK6] Embedding rows: {len(emb)}")

    if len(meta) != len(emb):
        raise ValueError(
            f"Row mismatch between metadata ({len(meta)}) "
            f"and embeddings ({len(emb)}). They must align 1:1.\n"
            f"Please re-run Week 5 pipeline."
        )

    if "text" not in meta.columns:
        raise ValueError(
            f"Metadata must contain a 'text' column for intent labeling.\n"
            f"Available columns: {list(meta.columns)}"
        )

    print("[WEEK6] ✓ Input validation successful!\n")
    return meta, emb


def label_intents(meta: pd.DataFrame) -> List[str]:
    """Generate intent labels using rule-based approach"""
    print("[WEEK6] Applying rule-based intent labeling...")
    intents = meta["text"].fillna("").astype(str).apply(rule_based_intent).tolist()
    print(f"[WEEK6] Generated {len(intents)} intent labels")
    return intents


def train_intent_classifier(X: np.ndarray, intents: List[str]):
    """
    Train a simple multi-class intent classifier on top of
    joint speech-text embeddings.
    """
    print("[WEEK6] Preparing data for intent classification...")
    y_raw = np.array(intents)

    # Keep all rows (including OTHER); you can filter later if needed.
    valid_idx = np.arange(len(y_raw))
    X = X[valid_idx]
    y_raw = y_raw[valid_idx]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Ensure we have at least 2 classes
    if len(le.classes_) < 2:
        raise ValueError(
            f"Intent labeling produced < 2 classes.\n"
            f"Please refine rule_based_intent or label data manually.\n"
            f"Found classes: {le.classes_}"
        )

    print(f"[WEEK6] Detected {len(le.classes_)} intent classes: {list(le.classes_)}")

    # Check class distribution for stratification
    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()
    
    print(f"[WEEK6] Class distribution:")
    for cls, cnt in zip(le.classes_, counts):
        print(f"[WEEK6]   {cls}: {cnt} samples")
    
    if min_samples < 2:
        print(f"[WEEK6] ⚠ Warning: Some classes have < 2 samples. Using random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"\n[WEEK6] Training classifier...")
    print(f"[WEEK6] Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="auto",
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("[WEEK6] Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val, y_pred, target_names=le.classes_, zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred)

    print("\n" + "=" * 65)
    print("WEEK 6 INTENT CLASSIFIER RESULTS")
    print("=" * 65)
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"Number of intent classes: {len(le.classes_)}")
    print(f"Intent classes: {list(le.classes_)}\n")
    print(f"Validation Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("=" * 65 + "\n")

    return clf, le, acc, report, cm


def save_intent_model(clf, label_encoder, feature_dim: int):
    """
    Persist everything needed for later inference.
    This is your 'intent-aware speech model' artifact.
    """
    model_pack = {
        "classifier": clf,
        "label_encoder": label_encoder,
        "feature_dim": feature_dim,
        "intent_classes": list(label_encoder.classes_),
        "model_type": "LogisticRegression",
        "week5_encoder_path": WEEK5_ENCODER_PATH,
        # Note: Week-5 encoder, scaler etc. are saved separately.
        # At inference:
        #  1) encode text/audio -> joint embedding (Week 5 encoder)
        #  2) feed embedding to this classifier for intent prediction.
    }
    joblib.dump(model_pack, INTENT_MODEL_PATH)
    print(f"[WEEK6] ✓ Intent model saved → {INTENT_MODEL_PATH}")
    print(f"[WEEK6] Model contains {len(label_encoder.classes_)} intent classes")


def week6_pipeline():
    """Main Week 6 pipeline execution"""
    # Capture all console output into a report file
    old_stdout = sys.stdout
    mystdout = StringIO()
    sys.stdout = mystdout

    try:
        print("\n" + "=" * 65)
        print("WEEK 6: INTENT DETECTION FINE-TUNING (ReDial)")
        print("=" * 65 + "\n")
        print("[WEEK6] Starting Week 6 pipeline...")
        print(f"[WEEK6] Output directory: {OUTPUT_DIR}\n")
        
        print("[WEEK6] Step 1: Loading Week-4 metadata + Week-5 embeddings...")
        meta, emb = load_inputs()
        X = emb.values.astype(np.float32)

        print(f"[WEEK6] Step 2: Dataset Summary:")
        print(f"[WEEK6]   - Total samples: {len(meta)}")
        print(f"[WEEK6]   - Embedding dimension: {X.shape[1]}")
        print(f"[WEEK6]   - Embedding shape: {X.shape}\n")

        print("[WEEK6] Step 3: Generating rule-based intent labels...")
        intents = label_intents(meta)

        # Show intent label distribution
        intent_series = pd.Series(intents)
        print("\n[WEEK6] Intent distribution (rule-based labels):")
        for intent, count in intent_series.value_counts().items():
            pct = (count / len(intent_series)) * 100
            print(f"[WEEK6]   {intent}: {count} ({pct:.1f}%)")
        print()

        print("[WEEK6] Step 4: Training intent classifier on joint embeddings...")
        clf, le, acc, report, cm = train_intent_classifier(X, intents)

        print("[WEEK6] Step 5: Saving intent-aware speech model...")
        save_intent_model(clf, le, X.shape[1])

        print("\n" + "=" * 65)
        print("WEEK 6 FINAL SUMMARY")
        print("=" * 65)
        print(f"Total samples processed: {len(meta)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Number of intents: {len(le.classes_)}")
        print(f"Intent classes: {', '.join(le.classes_)}")
        print(f"Validation accuracy: {acc:.4f}")
        print(f"\nObjective: Fine-tuned CRS intent classifier")
        print(f"Method: Logistic Regression on speech-text embeddings")
        print("\nOutput files:")
        print(f"  1. Model: {os.path.basename(INTENT_MODEL_PATH)}")
        print(f"  2. Report: {os.path.basename(INTENT_REPORT_PATH)}")
        print("=" * 65)
        
        print("\n[WEEK6] ✓✓✓ Week 6 pipeline completed successfully! ✓✓✓\n")

    except Exception as e:
        print("\n" + "=" * 65)
        print("ERROR: WEEK 6 PIPELINE FAILED")
        print("=" * 65)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}\n")
        
        import traceback
        print("Full traceback:")
        print("-" * 65)
        print(traceback.format_exc())
        print("-" * 65)
        
        print("\nTroubleshooting steps:")
        print("1. Make sure you ran Week 4 pipeline first")
        print("2. Make sure you ran Week 5 pipeline first")
        print("3. Check that all files exist in:", OUTPUT_DIR)
        print("\nRequired files:")
        print(f"  - {os.path.basename(TRAIN_META)}")
        print(f"  - {os.path.basename(TRAIN_EMB_PATH)}")
        print(f"  - {os.path.basename(WEEK5_ENCODER_PATH)} (optional)")
        
        raise
    finally:
        # Restore stdout and write report to file
        sys.stdout = old_stdout
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(INTENT_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(mystdout.getvalue())

        print(f"\n{'='*65}")
        print(f"[WEEK6] Training report saved → {INTENT_REPORT_PATH}")
        
        if os.path.exists(INTENT_MODEL_PATH):
            print(f"[WEEK6] Intent model saved → {INTENT_MODEL_PATH}")
            print(f"[WEEK6] ✓ Week 6 pipeline completed successfully!")
        else:
            print(f"[WEEK6] ⚠ Warning: Model file not created. Check report for errors.")
        print(f"{'='*65}\n")


# --------- INFERENCE HELPERS ---------
def load_intent_model():
    """
    Utility to load the Week 6 model from disk.
    Returns (classifier, label_encoder, model_info).
    """
    if not os.path.exists(INTENT_MODEL_PATH):
        raise FileNotFoundError(
            f"Intent model not found: {INTENT_MODEL_PATH}\n"
            f"Please run Week 6 pipeline first:\n"
            f"  python week6_intent_finetune.py"
        )
    
    pack = joblib.load(INTENT_MODEL_PATH)
    print(f"[WEEK6-INFERENCE] Loaded intent model with {len(pack['intent_classes'])} classes")
    print(f"[WEEK6-INFERENCE] Classes: {', '.join(pack['intent_classes'])}")
    return pack["classifier"], pack["label_encoder"], pack


def predict_intent_for_embedding(embedding: np.ndarray, verbose: bool = True) -> str:
    """
    Given a single joint embedding vector (same format as a row
    from speech_text_embeddings_train.csv), predict the intent label.
    
    Args:
        embedding: numpy array of shape (feature_dim,) or (1, feature_dim)
        verbose: whether to print prediction details
    
    Returns:
        Predicted intent label as string
    """
    clf, le, info = load_intent_model()
    
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    
    # Validate dimension
    if emb.shape[1] != info["feature_dim"]:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {info['feature_dim']}, "
            f"got {emb.shape[1]}"
        )
    
    pred = clf.predict(emb)[0]
    intent_label = le.inverse_transform([pred])[0]
    
    if verbose:
        # Get confidence scores
        proba = clf.predict_proba(emb)[0]
        confidence = proba[pred]
        
        print(f"[WEEK6-INFERENCE] Predicted intent: {intent_label}")
        print(f"[WEEK6-INFERENCE] Confidence: {confidence:.3f}")
        
        # Show top 3 predictions
        top_3_idx = np.argsort(proba)[-3:][::-1]
        print(f"[WEEK6-INFERENCE] Top 3 predictions:")
        for idx in top_3_idx:
            print(f"[WEEK6-INFERENCE]   {le.classes_[idx]}: {proba[idx]:.3f}")
    
    return intent_label


def predict_intent_batch(embeddings: np.ndarray, verbose: bool = True) -> List[str]:
    """
    Predict intents for multiple embeddings at once.
    
    Args:
        embeddings: numpy array of shape (n_samples, feature_dim)
        verbose: whether to print prediction details
    
    Returns:
        List of predicted intent labels
    """
    clf, le, info = load_intent_model()
    
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    # Validate dimension
    if embeddings.shape[1] != info["feature_dim"]:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {info['feature_dim']}, "
            f"got {embeddings.shape[1]}"
        )
    
    predictions = clf.predict(embeddings)
    intent_labels = le.inverse_transform(predictions).tolist()
    
    if verbose:
        print(f"[WEEK6-INFERENCE] Predicted intents for {len(intent_labels)} samples")
        
        # Show distribution
        intent_dist = pd.Series(intent_labels).value_counts()
        print(f"[WEEK6-INFERENCE] Intent distribution:")
        for intent, count in intent_dist.items():
            pct = (count / len(intent_labels)) * 100
            print(f"[WEEK6-INFERENCE]   {intent}: {count} ({pct:.1f}%)")
    
    return intent_labels


if __name__ == "__main__":
    week6_pipeline()