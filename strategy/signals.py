import numpy as np
from typing import Optional


class TradingSignals:
    """
    Model-driven signal generator.
    Pure Kronos predictions — no technical filters overriding the model.
    """

    def __init__(
        self,
        min_confidence: float = 0.005,
        min_return_threshold: float = 0.001,
    ):
        self.min_confidence = min_confidence
        self.min_return_threshold = min_return_threshold

    def generate_signals(
        self,
        predictions: dict[str, dict],
        stock_data: dict,
    ) -> list[dict]:
        """
        Generate ranked trading signals from Kronos predictions.
        Model decides everything — we just rank by predicted_return * confidence.
        """
        signals = []

        for sym, pred in predictions.items():
            predicted_return = pred.get("predicted_return", 0)
            confidence = pred.get("confidence", 0)
            current_close = pred.get("current_close", 0)
            predicted_close = pred.get("predicted_close", 0)

            # Skip if confidence too low
            if confidence < self.min_confidence:
                continue

            # Skip if prediction is essentially flat
            if abs(predicted_return) < self.min_return_threshold:
                continue

            # Model-driven action
            if predicted_return > 0:
                action = "buy"
            elif predicted_return < 0:
                action = "sell"
            else:
                action = "hold"

            # Score = predicted_return weighted by confidence
            score = predicted_return * confidence

            signals.append({
                "symbol": sym,
                "action": action,
                "score": score,
                "predicted_return": predicted_return,
                "confidence": confidence,
                "current_close": current_close,
                "predicted_close": predicted_close,
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals
