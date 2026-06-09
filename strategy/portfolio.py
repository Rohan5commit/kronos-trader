import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional


class Portfolio:
    """
    Paper trading portfolio tracker using SQLite.
    Manages positions, trades, cash, and P&L.
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
                    stop_loss REAL,
                    take_profit REAL
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

            # Initialize cash if not set
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
        self, symbol: str, price: float, shares: float, reason: str = "signal"
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
                stop_loss = price * 0.95
                take_profit = price * 1.15
                conn.execute(
                    "INSERT INTO positions (symbol, shares, avg_cost, entry_date, stop_loss, take_profit) VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol, shares, price, datetime.now().isoformat(), stop_loss, take_profit),
                )

            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, "buy", shares, price, cost, datetime.now().isoformat(), reason),
            )

            new_cash = cash - cost
            conn.execute(
                "UPDATE state SET value = ? WHERE key = 'cash'", (str(new_cash),)
            )

        print(f"  BOUGHT {shares} shares of {symbol} @ ${price:.2f} (${cost:.2f})")
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

        print(f"  SOLD {shares} shares of {symbol} @ ${price:.2f} (P&L: ${pnl:+.2f}) - {reason}")
        return True

    def check_stop_loss_take_profit(
        self, current_prices: dict[str, float]
    ) -> list[str]:
        """Check and execute stop-loss / take-profit triggers."""
        actions = []
        positions = self.get_positions()

        for sym, pos in positions.items():
            price = current_prices.get(sym)
            if price is None:
                continue

            if price <= pos["stop_loss"]:
                self.sell(sym, price, reason="stop_loss")
                actions.append(f"STOP LOSS: {sym} @ ${price:.2f}")
            elif price >= pos["take_profit"]:
                self.sell(sym, price, reason="take_profit")
                actions.append(f"TAKE PROFIT: {sym} @ ${price:.2f}")

        return actions

    def rebalance(
        self,
        signals: list[dict],
        current_prices: dict[str, float],
        max_positions: int = 50,
        position_size_pct: float = 0.02,
    ) -> list[str]:
        """
        Execute rebalancing based on signals.
        Sells stocks not in top signals, buys new top signals.
        """
        actions = []
        total_value = self.get_total_value(current_prices)
        cash = self.get_cash()

        # Determine which stocks to hold (top N by score)
        buy_signals = [s for s in signals if s["action"] == "buy"][:max_positions]
        target_syms = {s["symbol"] for s in buy_signals}

        # Sell positions not in target
        positions = self.get_positions()
        for sym in positions:
            if sym not in target_syms:
                price = current_prices.get(sym, positions[sym]["avg_cost"])
                self.sell(sym, price, reason="rebalance_out")
                actions.append(f"SELL: {sym}")

        # Buy new positions
        position_size = total_value * position_size_pct
        for sig in buy_signals:
            sym = sig["symbol"]
            price = current_prices.get(sym, sig["current_close"])
            current_pos = positions.get(sym, {}).get("shares", 0)

            if current_pos > 0:
                continue  # Already holding

            if cash < position_size:
                break  # Not enough cash

            shares = int(position_size / price)
            if shares > 0:
                self.buy(sym, price, shares, reason="rebalance_in")
                actions.append(f"BUY: {sym} x{shares}")
                cash = self.get_cash()

        return actions

    def snapshot(self, current_prices: dict[str, float]) -> dict:
        """Take daily portfolio snapshot."""
        cash = self.get_cash()
        equity = self.get_total_equity(current_prices)
        total = cash + equity
        positions = self.get_positions()

        # Get yesterday's snapshot for daily P&L
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

        # Top winners and losers
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
