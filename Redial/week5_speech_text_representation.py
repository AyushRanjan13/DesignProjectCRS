"""
WEEK 5 — Speech-Text Representation (ReDial)
Goal:
- Generate text + audio embeddings
- Train joint speech-text encoder using contrastive learning + masked prediction
- Save aligned embeddings for TRAIN only
- Save encoder + training report
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import librosa
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR


# Week-4 enriched metadata
TRAIN_META = os.path.join(OUTPUT_DIR, "metadata_with_acoustic_emotion.csv")

REPORT_FILE = os.path.join(OUTPUT_DIR, "week5_report.txt")
MODEL_PATH = os.path.join(OUTPUT_DIR, "week5_encoder.joblib")

# Output embeddings for TRAIN only
TRAIN_EMB_PATH = os.path.join(OUTPUT_DIR, "speech_text_embeddings_train.csv")


def extract_audio_embedding(wav, n_mels=64):
    """Return mean mel-spectrogram embedding"""
    try:
        y, sr = librosa.load(wav, sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel)
        return np.mean(mel_db, axis=1)
    except:
        return np.zeros(n_mels)


class DualAligner(nn.Module):
    def __init__(self, text_dim, audio_dim, proj_dim=256):
        super().__init__()

        self.text_net = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        
        # Reconstruction head for masked prediction
        self.text_reconstruct = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, text_dim)
        )
        
        self.audio_reconstruct = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, audio_dim)
        )

    def forward_text(self, x):
        return self.text_net(x)

    def forward_audio(self, x):
        return self.audio_net(x)
    
    def reconstruct_text(self, proj):
        return self.text_reconstruct(proj)
    
    def reconstruct_audio(self, proj):
        return self.audio_reconstruct(proj)


class PairDataset(Dataset):
    def __init__(self, text_emb, audio_emb):
        self.text = text_emb
        self.audio = audio_emb

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text[idx], dtype=torch.float32),
            torch.tensor(self.audio[idx], dtype=torch.float32)
        )


def contrastive_loss(x, y, temperature=0.05):
    """NT-Xent Loss"""
    x = nn.functional.normalize(x, dim=-1)
    y = nn.functional.normalize(y, dim=-1)

    logits = x @ y.T / temperature
    labels = torch.arange(len(x)).to(x.device)
    return nn.CrossEntropyLoss()(logits, labels)


def masked_prediction_loss(original_emb, projected_emb, reconstructed_emb, mask_ratio=0.15):
    """
    Masked prediction loss - randomly mask embeddings and predict them
    """
    batch_size = original_emb.shape[0]
    
    # Create random mask
    mask = torch.rand(batch_size, device=original_emb.device) < mask_ratio
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=original_emb.device)
    
    # Reconstruction loss for masked samples
    reconstruction_loss = nn.MSELoss()(
        reconstructed_emb[mask], 
        original_emb[mask]
    )
    
    return reconstruction_loss


def week5_pipeline(epochs=8, batch_size=32, proj_dim=256, contrastive_weight=1.0, masked_weight=0.5):

    old_stdout = sys.stdout
    from io import StringIO
    sys.stdout = mystdout = StringIO()

    print("\n========= WEEK 5: SPEECH-TEXT REPRESENTATION (ReDial) =========\n")
    print("[WEEK5] Training with Contrastive + Masked Prediction Objectives")

    # Load metadata
    if not os.path.exists(TRAIN_META):
        print("Training metadata missing ->", TRAIN_META)
        sys.stdout = old_stdout
        raise FileNotFoundError(TRAIN_META)

    df = pd.read_csv(TRAIN_META)
    print(f"[WEEK5] Loaded training metadata -> {len(df)} samples")

    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # ------------------------------------------------
    # TEXT embeddings
    # ------------------------------------------------
    print("[WEEK5] Generating text embeddings...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_emb = text_model.encode(df["text"].tolist(), show_progress_bar=True)

    # ------------------------------------------------
    # AUDIO embeddings
    # ------------------------------------------------
    print("[WEEK5] Generating audio embeddings...")
    audio_emb = []
    for p in tqdm(df["audio_path"]):
        audio_emb.append(extract_audio_embedding(p))

    text_emb = np.array(text_emb)
    audio_emb = np.array(audio_emb)

    scaler = StandardScaler()
    audio_emb = scaler.fit_transform(audio_emb)

    dataset = PairDataset(text_emb, audio_emb)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    text_dim = text_emb.shape[1]
    audio_dim = audio_emb.shape[1]

    aligner = DualAligner(text_dim, audio_dim, proj_dim)
    opt = torch.optim.Adam(aligner.parameters(), lr=1e-3)

    # ------------------------------------------------
    # Train contrastive joint encoder with masked prediction
    # ------------------------------------------------
    print(f"\n[WEEK5] Training for {epochs} epochs...")
    print(f"[WEEK5] Loss weights: Contrastive={contrastive_weight}, Masked={masked_weight}\n")
    
    aligner.train()
    for epoch in range(epochs):
        epoch_losses = {
            'total': [],
            'contrastive': [],
            'masked_text': [],
            'masked_audio': []
        }
        
        for t, a in loader:
            opt.zero_grad()
            
            # Forward pass
            t_proj = aligner.forward_text(t)
            a_proj = aligner.forward_audio(a)
            
            # Contrastive loss
            loss_contrastive = contrastive_loss(t_proj, a_proj)
            
            # Masked prediction losses
            t_recon = aligner.reconstruct_text(t_proj)
            a_recon = aligner.reconstruct_audio(a_proj)
            
            loss_masked_text = masked_prediction_loss(t, t_proj, t_recon, mask_ratio=0.15)
            loss_masked_audio = masked_prediction_loss(a, a_proj, a_recon, mask_ratio=0.15)
            
            # Combined loss
            total_loss = (
                contrastive_weight * loss_contrastive + 
                masked_weight * (loss_masked_text + loss_masked_audio)
            )
            
            total_loss.backward()
            opt.step()
            
            # Track losses
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['contrastive'].append(loss_contrastive.item())
            epoch_losses['masked_text'].append(loss_masked_text.item())
            epoch_losses['masked_audio'].append(loss_masked_audio.item())

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total Loss: {np.mean(epoch_losses['total']):.4f}")
        print(f"  Contrastive: {np.mean(epoch_losses['contrastive']):.4f}")
        print(f"  Masked Text: {np.mean(epoch_losses['masked_text']):.4f}")
        print(f"  Masked Audio: {np.mean(epoch_losses['masked_audio']):.4f}")

    # ------------------------------------------------
    # Save model
    # ------------------------------------------------
    model_data = {
        'state_dict': aligner.state_dict(),
        'scaler': scaler,
        'text_dim': text_dim,
        'audio_dim': audio_dim,
        'proj_dim': proj_dim
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n[WEEK5] Aligner model saved -> {MODEL_PATH}")

    # ------------------------------------------------
    # Generate aligned embeddings for TRAIN
    # ------------------------------------------------
    aligner.eval()
    with torch.no_grad():
        proj_text = aligner.forward_text(
            torch.tensor(text_emb, dtype=torch.float32)
        ).numpy()

        proj_audio = aligner.forward_audio(
            torch.tensor(audio_emb, dtype=torch.float32)
        ).numpy()

    # Named columns (INSPIRED-style)
    text_col_names = [f"text_emb_{i}" for i in range(proj_text.shape[1])]
    audio_col_names = [f"audio_emb_{i}" for i in range(proj_audio.shape[1])]
    all_col_names = text_col_names + audio_col_names

    combined_emb = np.hstack([proj_text, proj_audio])
    out = pd.DataFrame(combined_emb, columns=all_col_names)
    out.to_csv(TRAIN_EMB_PATH, index=False)

    print(f"[WEEK5] Train embeddings saved -> {TRAIN_EMB_PATH}")
    print(f"[WEEK5] Embedding shape: {combined_emb.shape}")
    print(
        f"[WEEK5] Columns: {len(text_col_names)} text + "
        f"{len(audio_col_names)} audio = {len(all_col_names)} total"
    )

    # Summary statistics
    print(f"\n[WEEK5] ===== TRAINING SUMMARY =====")
    print(f"[WEEK5] Dataset size: {len(df)} samples")
    print(f"[WEEK5] Text embedding dim: {text_dim}")
    print(f"[WEEK5] Audio embedding dim: {audio_dim}")
    print(f"[WEEK5] Projection dim: {proj_dim}")
    print(f"[WEEK5] Training epochs: {epochs}")
    print(f"[WEEK5] Batch size: {batch_size}")
    print(f"[WEEK5] Objectives: Contrastive Learning + Masked Prediction")

    # ------------------------------------------------
    # Save report log
    # ------------------------------------------------
    sys.stdout = old_stdout
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(mystdout.getvalue())

    print(f"[WEEK5] Training logs saved -> {REPORT_FILE}")
    print(f"[WEEK5] Week 5 pipeline completed successfully!")


if __name__ == "__main__":
    week5_pipeline()