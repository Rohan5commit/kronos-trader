import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Add Kronos repo to path
KRONOS_DIR = Path(__file__).parent.parent / "kronos_repo"


class KronosBatchPredictor:
    """Batch inference engine for Kronos on GPU."""

    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = Path(model_cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Kronos tokenizer and model from cache or HuggingFace."""
        sys.path.insert(0, str(KRONOS_DIR))

        from model import Kronos, KronosTokenizer

        tokenizer_path = self.model_cache_dir / "tokenizer"
        model_path = self.model_cache_dir / "model"

        if tokenizer_path.exists() and model_path.exists():
            print(f"Loading cached tokenizer from {tokenizer_path}")
            self.tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
        else:
            print("Downloading tokenizer from HuggingFace...")
            self.tokenizer = KronosTokenizer.from_pretrained(
                "NeoQuasar/Kronos-Tokenizer-base"
            )
            self.tokenizer.save_pretrained(str(tokenizer_path))

        if model_path.exists():
            print(f"Loading cached model from {model_path}")
            self.model = Kronos.from_pretrained(str(model_path))
        else:
            print("Downloading model from HuggingFace...")
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            self.model.save_pretrained(str(model_path))

        self.model = self.model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

    def predict_batch(
        self,
        stock_data: dict[str, pd.DataFrame],
        pred_len: int = 10,
        context_length: int = 512,
        batch_size: int = 32,
        T: float = 0.8,
        top_p: float = 0.9,
    ) -> dict[str, dict]:
        """
        Run batch prediction on multiple stocks.

        Returns dict: {symbol: {"predicted_close": float, "predicted_return": float,
                                "confidence": float, "current_close": float}}
        """
        from model import KronosPredictor

        predictor = KronosPredictor(
            self.model, self.tokenizer, device=str(self.device),
            max_context=context_length
        )

        results = {}
        symbols = list(stock_data.keys())

        for i in range(0, len(symbols), batch_size):
            batch_syms = symbols[i : i + batch_size]
            batch_dfs = [stock_data[s] for s in batch_syms]

            for j, sym in enumerate(batch_syms):
                df = batch_dfs[j]
                try:
                    if len(df) < context_length:
                        continue

                    x_df = df.tail(context_length + pred_len)
                    x_hist = x_df.iloc[:context_length][
                        ["open", "high", "low", "close", "volume"]
                    ].copy()
                    x_hist["amount"] = 0.0

                    x_timestamp = x_df.iloc[:context_length]["timestamp"]
                    y_timestamp = x_df.iloc[
                        context_length : context_length + pred_len
                    ]["timestamp"]

                    if len(y_timestamp) < pred_len:
                        continue

                    pred_df = predictor.predict(
                        df=x_hist,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=T,
                        top_p=top_p,
                        sample_count=1,
                    )

                    current_close = float(df.iloc[-1]["close"])
                    predicted_close = float(pred_df["close"].iloc[-1])
                    avg_predicted_close = float(pred_df["close"].mean())

                    predicted_return = (predicted_close - current_close) / current_close
                    avg_return = (avg_predicted_close - current_close) / current_close

                    # Confidence based on prediction spread
                    pred_std = float(pred_df["close"].std())
                    confidence = abs(predicted_return) / (pred_std / current_close + 1e-8)

                    results[sym] = {
                        "current_close": current_close,
                        "predicted_close": predicted_close,
                        "avg_predicted_close": avg_predicted_close,
                        "predicted_return": predicted_return,
                        "avg_return": avg_return,
                        "confidence": confidence,
                        "pred_std_pct": pred_std / current_close,
                    }

                except Exception as e:
                    print(f"  Prediction error for {sym}: {e}")
                    continue

            print(f"  Predicted batch {i+1}-{min(i+batch_size, len(symbols))}/{len(symbols)}")

        return results
