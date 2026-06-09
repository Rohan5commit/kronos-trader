import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional


class Portfolio:
    """
    Fully model-driven paper trading portfolio.
    Buy/sell decisions come entirely from Kronos predictions.
    No hard-coded stop-loss, take-profit, or rebalancing.
    """

    def __init__(self, db_path: str, initial_cash: float = 100_000.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_cash = initial_cash
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    shares REAL DEFAULT 0,
                    avg_cost REAL DEFAULT 0,
                    entry_date TEXT,
                    last_prediction REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    action TEXT,
                    shares REAL,
                    price REAL,
                    total REAL,
                    timestamp TEXT,
                    reason TEXT
                );
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE,
                    cash REAL,
                    equity REAL,
                    total_value REAL,
                    positions_json TEXT,
                    daily_pnl REAL,
                    cumulative_pnl REAL
                );
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            existing = conn.execute(
                "SELECT value FROM state WHERE key = 'cash'"
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO state (key, value) VALUES ('cash', ?)",
                    (str(self.initial_cash),),
                )

    def get_cash(self) -> float:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT value FROM state WHERE key = 'cash'"
            ).fetchone()
            return float(row[0]) if row else self.initial_cash

    def get_positions(self) -> dict:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM positions WHERE shares > 0").fetchall()
            return {r["symbol"]: dict(r) for r in rows}

    def get_total_equity(self, current_prices: dict[str, float]) -> float:
        positions = self.get_positions()
        equity = 0
        for sym, pos in positions.items():
            price = current_prices.get(sym, pos["avg_cost"])
            equity += pos["shares"] * price
        return equity

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        return self.get_cash() + self.get_total_equity(current_prices)

    def buy(
        self, symbol: str, price: float, shares: float, reason: str = "signal",
        predicted_return: float = 0.0,
    ) -> bool:
        cost = price * shares
        cash = self.get_cash()
        if cost > cash:
            shares = int(cash / price)
            if shares <= 0:
                return False
            cost = price * shares

        with sqlite3.connect(str(self.db_path)) as conn:
            existing = conn.execute(
                "SELECT shares, avg_cost FROM positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()

            if existing:
                old_shares, old_avg = existing
                new_shares = old_shares + shares
                new_avg = (old_shares * old_avg + cost) / new_shares
                conn.execute(
                    "UPDATE positions SET shares = ?, avg_cost = ? WHERE symbol = ?",
                    (new_shares, new_avg, symbol),
                )
            else:
                conn.execute(
                    "INSERT INTO positions (symbol, shares, avg_cost, entry_date, last_prediction) VALUES (?, ?, ?, ?, ?)",
                    (symbol, shares, price, datetime.now().isoformat(), predicted_return),
                )

            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, "buy", shares, price, cost, datetime.now().isoformat(), reason),
            )

            new_cash = cash - cost
            conn.execute(
                "UPDATE state SET value = ? WHERE key = 'cash'", (str(new_cash),)
            )

        print(f"  BUY  {shares:>6} x {symbol:<8} @ ${price:>8.2f}  (${cost:>10.2f})  pred={predicted_return:+.2%}")
        return True

    def sell(self, symbol: str, price: float, reason: str = "signal") -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            pos = conn.execute(
                "SELECT shares, avg_cost FROM positions WHERE symbol = ? AND shares > 0",
                (symbol,),
            ).fetchone()
            if not pos:
                return False

            shares, avg_cost = pos
            proceeds = price * shares
            pnl = (price - avg_cost) * shares
            pnl_pct = (price - avg_cost) / avg_cost * 100

            conn.execute(
                "UPDATE positions SET shares = 0 WHERE symbol = ?", (symbol,)
            )
            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, "sell", shares, price, proceeds, datetime.now().isoformat(), reason),
            )

            cash = self.get_cash()
            new_cash = cash + proceeds
            conn.execute(
                "UPDATE state SET value = ? WHERE key = 'cash'", (str(new_cash),)
            )

        print(f"  SELL {shares:>6} x {symbol:<8} @ ${price:>8.2f}  (P&L: ${pnl:>+10.2f} {pnl_pct:>+6.1f}%)  {reason}")
        return True

    def manage_positions(
        self,
        predictions: dict[str, dict],
        current_prices: dict[str, float],
        max_positions: int = 50,
        min_confidence: float = 0.005,
    ) -> list[str]:
        """
        Model-driven position management:
        1. Sell held positions where model predicts negative return or low confidence
        2. Buy top predicted stocks not yet held
        Position size scaled by confidence.
        """
        actions = []
        positions = self.get_positions()
        cash = self.get_cash()
        total_value = self.get_total_value(current_prices)

        # ---- STEP 1: SELL ----
        # Sell if model predicts return < 0 or confidence too low
        for sym, pos in list(positions.items()):
            price = current_prices.get(sym)
            if price is None:
                continue

            pred = predictions.get(sym)
            if pred is None:
                # No prediction available — model doesn't want it anymore
                self.sell(sym, price, reason="no_prediction")
                actions.append(f"SELL {sym} (no prediction)")
                continue

            predicted_return = pred.get("predicted_return", 0)
            confidence = pred.get("confidence", 0)

            # Model says sell: negative return or too uncertain
            if predicted_return < 0 or confidence < min_confidence:
                self.sell(sym, price, reason=f"model_sell: ret={predicted_return:+.2%} conf={confidence:.3f}")
                actions.append(f"SELL {sym} (ret={predicted_return:+.2%})")

        # ---- STEP 2: BUY ----
        # Rank all predictions by score, buy top N not yet held
        buy_candidates = []
        for sym, pred in predictions.items():
            if sym in positions and positions[sym].get("shares", 0) > 0:
                continue  # Already holding

            predicted_return = pred.get("predicted_return", 0)
            confidence = pred.get("confidence", 0)

            if predicted_return <= 0:
                continue  # Model doesn't predict upside
            if confidence < min_confidence:
                continue  # Too uncertain

            # Score = return * confidence (simple model-driven ranking)
            score = predicted_return * confidence
            buy_candidates.append((sym, pred, score))

        buy_candidates.sort(key=lambda x: x[2], reverse=True)

        # How many more positions can we take?
        current_positions = len([p for p in self.get_positions().values() if p["shares"] > 0])
        slots = max_positions - current_positions

        cash = self.get_cash()
        for sym, pred, score in buy_candidates[:slots]:
            if cash < 100:
                break

            price = current_prices.get(sym, pred.get("current_close", 0))
            if price <= 0:
                continue

            confidence = pred.get("confidence", 0)
            predicted_return = pred.get("predicted_return", 0)

            # Position size scaled by confidence: 1%-5% of portfolio
            confidence_multiplier = min(confidence / 0.05, 1.0)
            position_pct = 0.01 + 0.04 * confidence_multiplier  # 1% to 5%
            position_value = total_value * position_pct
            position_value = min(position_value, cash * 0.95)  # Never use more than 95% of cash

            shares = int(position_value / price)
            if shares > 0:
                if self.buy(sym, price, shares, reason="model_buy", predicted_return=predicted_return):
                    actions.append(f"BUY {sym} x{shares} (ret={predicted_return:+.2%}, conf={confidence:.3f})")
                    cash = self.get_cash()

        return actions

    def snapshot(self, current_prices: dict[str, float]) -> dict:
        """Take daily portfolio snapshot."""
        cash = self.get_cash()
        equity = self.get_total_equity(current_prices)
        total = cash + equity
        positions = self.get_positions()

        with sqlite3.connect(str(self.db_path)) as conn:
            prev = conn.execute(
                "SELECT total_value FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
            ).fetchone()
            prev_total = float(prev[0]) if prev else self.initial_cash

        daily_pnl = total - prev_total
        cumulative_pnl = total - self.initial_cash

        today = date.today().isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_snapshots
                   (date, cash, equity, total_value, positions_json, daily_pnl, cumulative_pnl)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    today, cash, equity, total,
                    json.dumps({s: dict(p) for s, p in positions.items()}),
                    daily_pnl, cumulative_pnl,
                ),
            )

        return {
            "date": today,
            "cash": cash,
            "equity": equity,
            "total_value": total,
            "daily_pnl": daily_pnl,
            "cumulative_pnl": cumulative_pnl,
            "num_positions": len([p for p in positions.values() if p["shares"] > 0]),
            "positions": positions,
        }

    def get_summary(self, current_prices: dict[str, float]) -> dict:
        """Get full portfolio summary for email."""
        snap = self.snapshot(current_prices)
        positions = self.get_positions()

        pos_pnl = []
        for sym, pos in positions.items():
            if pos["shares"] <= 0:
                continue
            price = current_prices.get(sym, pos["avg_cost"])
            pnl = (price - pos["avg_cost"]) * pos["shares"]
            pnl_pct = (price - pos["avg_cost"]) / pos["avg_cost"] * 100
            pos_pnl.append({
                "symbol": sym,
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
                "current_price": price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })

        pos_pnl.sort(key=lambda x: x["pnl"], reverse=True)

        with sqlite3.connect(str(self.db_path)) as conn:
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            recent_trades = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()

        return {
            **snap,
            "top_winners": pos_pnl[:5],
            "top_losers": pos_pnl[-5:],
            "total_trades": total_trades,
            "recent_trades": recent_trades,
        }
