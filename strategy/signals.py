import numpy as np
from typing import Optional


class TradingSignals:
    """
    Autonomous hedge fund signal generator.

    Combines Kronos predictions with technical filters to generate
    buy/sell/hold signals. Goal: maximize risk-adjusted returns.
    """

    def __init__(
        self,
        min_confidence: float = 0.005,
        min_return_threshold: float = 0.005,
        max_volatility: float = 0.08,
    ):
        self.min_confidence = min_confidence
        self.min_return_threshold = min_return_threshold
        self.max_volatility = max_volatility

    def generate_signals(
        self,
        predictions: dict[str, dict],
        stock_data: dict,
    ) -> list[dict]:
        """
        Generate ranked trading signals from Kronos predictions.

        Returns sorted list of {symbol, action, score, ...}
        """
        signals = []

        for sym, pred in predictions.items():
            predicted_return = pred.get("predicted_return", 0)
            confidence = pred.get("confidence", 0)
            current_close = pred.get("current_close", 0)
            pred_std_pct = pred.get("pred_std_pct", 0)

            # Skip low-confidence predictions
            if confidence < self.min_confidence:
                continue

            # Skip extremely volatile predictions
            if pred_std_pct > self.max_volatility:
                continue

            # Calculate composite score
            # Weight: 60% predicted return, 25% confidence, 15% inverse volatility
            vol_score = 1.0 / (pred_std_pct + 0.01)
            score = (
                0.60 * predicted_return
                + 0.25 * np.tanh(confidence)
                + 0.15 * np.tanh(vol_score)
            )

            # Additional filters from historical data
            hist_data = stock_data.get(sym)
            if hist_data is not None and len(hist_data) >= 20:
                closes = hist_data["close"].values

                # Momentum filter: prefer stocks in uptrend
                sma_20 = np.mean(closes[-20:])
                sma_5 = np.mean(closes[-5:])
                momentum = (sma_5 - sma_20) / sma_20

                # Mean reversion filter: avoid overextended
                pct_from_high = (closes[-1] - np.max(closes[-50:])) / np.max(
                    closes[-50:]
                )

                # Boost score for trending, penalize for overextended
                if momentum > 0:
                    score *= 1.2
                if pct_from_high > 0.05:
                    score *= 0.8  # Near 52-week high, lower conviction

            action = "buy" if predicted_return > self.min_return_threshold else (
                "sell" if predicted_return < -self.min_return_threshold else "hold"
            )

            signals.append({
                "symbol": sym,
                "action": action,
                "score": score,
                "predicted_return": predicted_return,
                "confidence": confidence,
                "current_close": current_close,
                "predicted_close": pred.get("predicted_close", 0),
            })

        signals.sort(key=lambda x: x["score"], reverse=True)
        return signals
