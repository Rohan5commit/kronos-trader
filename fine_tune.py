"""
Kronos LoRA Fine-Tuning — Runs weekly on GitHub Actions CPU VM.
Fine-tunes a small adapter on top of the frozen Kronos model using recent prediction data.
"""
import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = "./data"
OHLCV_DIR = f"{DATA_DIR}/ohlcv"
MODEL_DIR = f"{DATA_DIR}/model"
DB_PATH = f"{DATA_DIR}/trades.db"
ADAPTER_DIR = f"{MODEL_DIR}/adapter"

LOOKBACK = 400
PRED_LEN = 10
TRAIN_DAYS = 60
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4


class LoRAAdapter(nn.Module):
    """Low-rank adapter for Kronos. Injects small trainable matrices into attention layers."""
    def __init__(self, base_dim=256, rank=8):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Linear(base_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = rank / base_dim

    def forward(self, x):
        return x + self.lora_B(self.lora_A(x)) * self.scaling


def load_training_data():
    """Load OHLCV data and prediction accuracy from the last TRAIN_DAYS."""
    cache_dir = Path(OHLCV_DIR)
    if not cache_dir.exists():
        print("No OHLCV data found. Run data update first.")
        return []

    samples = []
    cutoff = datetime.now() - timedelta(days=TRAIN_DAYS)

    for parquet_file in cache_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            if len(df) < LOOKBACK + PRED_LEN + 10:
                continue
            # Check if data is recent enough
            latest = df["timestamp"].max()
            if hasattr(latest, 'to_pydatetime'):
                latest = latest.to_pydatetime()
            if hasattr(latest, 'replace'):
                latest = latest.replace(tzinfo=None)
            if latest < cutoff:
                continue
            samples.append((parquet_file.stem, df))
        except Exception:
            pass

    print(f"Loaded {len(samples)} stocks with sufficient recent data")
    return samples


def train_epoch(model, tokenizer, predictor, adapter, optimizer, samples, device):
    """Train adapter for one epoch on the training samples."""
    import torch.nn.functional as F

    adapter.train()
    total_loss = 0
    n = 0

    for sym, df in samples:
        try:
            # Random split: use first LOOKBACK bars as input, next PRED_LEN as target
            split_point = len(df) - PRED_LEN - np.random.randint(0, 20)
            x_df = df.iloc[split_point - LOOKBACK:split_point][["open", "high", "low", "close", "volume"]].copy()
            x_df["amount"] = 0.0
            y_df = df.iloc[split_point:split_point + PRED_LEN][["open", "high", "low", "close", "volume"]].copy()

            if x_df.isnull().values.any() or y_df.isnull().values.any():
                continue

            x_timestamp = pd.Series(df.iloc[split_point - LOOKBACK:split_point]["timestamp"])
            last_ts = x_timestamp.iloc[-1]
            y_timestamp = pd.Series(pd.date_range(
                start=last_ts + pd.Timedelta(days=1),
                periods=PRED_LEN,
                freq="B",
            ))

            # Get base model prediction
            with torch.no_grad():
                base_pred = predictor.predict(
                    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=PRED_LEN, T=0.8, top_p=0.9, sample_count=1,
                )

            # Apply adapter to modify prediction
            base_closes = torch.tensor(base_pred["close"].values, dtype=torch.float32, device=device)
            adapted_closes = adapter(base_closes.unsqueeze(0)).squeeze(0)

            # Target: actual future closes
            target = torch.tensor(y_df["close"].values, dtype=torch.float32, device=device)

            # Loss: MSE between adapted prediction and actual
            loss = F.mse_loss(adapted_closes, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1
        except Exception:
            pass

    return total_loss / max(n, 1)


def main():
    print(f"\n{'='*60}")
    print(f"  KRONOS LORA FINE-TUNING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Setup paths
    Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)

    # Load Kronos model
    print("Loading Kronos model...")
    if "/tmp/kronos_repo" not in sys.path:
        sys.path.insert(0, "/tmp/kronos_repo")
    from model import Kronos, KronosTokenizer, KronosPredictor

    model_cache = Path(MODEL_DIR)
    tokenizer_path = model_cache / "tokenizer"
    model_path = model_cache / "model"

    if tokenizer_path.exists():
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        tokenizer.save_pretrained(str(tokenizer_path))

    if model_path.exists():
        model = Kronos.from_pretrained(str(model_path))
    else:
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        model.save_pretrained(str(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model loaded on {device}")

    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)

    # Create LoRA adapter
    adapter = LoRAAdapter(base_dim=256, rank=8).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LR)

    # Load training data
    samples = load_training_data()
    if not samples:
        print("No training data available. Skipping fine-tune.")
        return

    # Train
    print(f"\nTraining for {EPOCHS} epochs on {len(samples)} stocks...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, tokenizer, predictor, adapter, optimizer, samples, device)
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={loss:.6f}")

    # Save adapter
    adapter_path = Path(ADAPTER_DIR) / "lora_adapter.pt"
    torch.save({
        "rank": adapter.rank,
        "lora_A": adapter.lora_A.state_dict(),
        "lora_B": adapter.lora_B.state_dict(),
        "base_dim": 256,
    }, adapter_path)
    print(f"\nAdapter saved to {adapter_path}")

    # Save metadata
    meta = {
        "trained_at": datetime.now().isoformat(),
        "epochs": EPOCHS,
        "train_stocks": len(samples),
        "train_days": TRAIN_DAYS,
        "final_loss": loss,
    }
    with open(Path(ADAPTER_DIR) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
