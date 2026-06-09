"""
Kronos Paper Trading Engine — Modal Serverless
Runs 2x/day on T4 GPU, fetches 1000 US stocks, predicts with Kronos, trades autonomously.
"""
import os
import json
import sys
import time
import sqlite3
import smtplib
import requests
import numpy as np
import pandas as pd
import modal
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =============================================================================
# Modal App Setup
# =============================================================================
app = modal.App("kronos-trader")

# Container image: PyTorch + all deps + Kronos repo
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy",
        "pandas",
        "einops==0.8.1",
        "huggingface_hub==0.33.1",
        "matplotlib==3.9.3",
        "tqdm==4.67.1",
        "safetensors==0.6.2",
        "requests",
        "pyarrow",
    )
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/shiyu-coder/Kronos.git /root/kronos_repo"
    )
)

# Persistent volume for model weights + OHLCV cache + trade database
vol = modal.Volume.from_name("kronos-data", create_if_missing=True)

VOLUME_PATH = "/kronos-data"
MODEL_CACHE = f"{VOLUME_PATH}/model"
DATA_CACHE = f"{VOLUME_PATH}/ohlcv"
DB_PATH = f"{VOLUME_PATH}/trades.db"


# =============================================================================
# Core Trading Function
# =============================================================================
@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    gpu="T4",
    timeout=1800,  # 30 min max
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
        modal.Secret.from_name("kronos-email", required=False),
    ],
    memory=16384,  # 16GB RAM
)
def run_trading_cycle():
    """Main trading cycle — fetches data, runs inference, trades, emails report."""
    print(f"\n{'='*60}")
    print(f"  KRONOS TRADING ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    import torch

    # =========================================================================
    # 1. LOAD API KEYS
    # =========================================================================
    keys_raw = os.environ.get("TWELVE_DATA_KEYS", "[]")
    API_KEYS = json.loads(keys_raw) if keys_raw else []
    if not API_KEYS:
        print("ERROR: No Twelve Data API keys found in secrets")
        return
    print(f"Loaded {len(API_KEYS)} API keys")

    # =========================================================================
    # 2. FETCH STOCK UNIVERSE
    # =========================================================================
    stock_list_path = Path(VOLUME_PATH) / "stocks.json"
    if stock_list_path.exists():
        with open(stock_list_path) as f:
            all_symbols = json.load(f)
        print(f"Loaded {len(all_symbols)} cached symbols")
    else:
        print("Fetching stock universe from Twelve Data...")
        all_symbols = _fetch_stock_universe(API_KEYS)
        stock_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stock_list_path, "w") as f:
            json.dump(all_symbols, f)
        print(f"Cached {len(all_symbols)} symbols")

    # =========================================================================
    # 3. FETCH OHLCV DATA (incremental)
    # =========================================================================
    data_cache_dir = Path(DATA_CACHE)
    data_cache_dir.mkdir(parents=True, exist_ok=True)

    cached_data = _load_cached_data(data_cache_dir)
    symbols_to_fetch = [s for s in all_symbols if s not in cached_data]
    symbols_to_update = [s for s in all_symbols if s in cached_data]

    print(f"\nData status: {len(cached_data)} cached, {len(symbols_to_fetch)} new, {len(symbols_to_update)} to update")

    # Fetch new stocks (full history)
    if symbols_to_fetch:
        print(f"\nFetching {len(symbols_to_fetch)} new stocks...")
        new_data = _fetch_ohlcv_batch(API_KEYS, symbols_to_fetch, outputsize=600)
        for sym, df in new_data.items():
            df.to_parquet(data_cache_dir / f"{sym}.parquet", index=False)
            cached_data[sym] = df

    # Update existing stocks (latest bars only)
    if symbols_to_update:
        print(f"\nUpdating {len(symbols_to_update)} stocks with latest bars...")
        _update_latest_bars(API_KEYS, symbols_to_update, cached_data, data_cache_dir)

    # Filter to stocks with enough data
    valid_data = {
        s: df for s, df in cached_data.items()
        if len(df) >= 520  # Need context_length (512) + some buffer
    }
    print(f"\nValid stocks for prediction: {len(valid_data)}")

    vol.commit()

    # =========================================================================
    # 4. LOAD KRONOS MODEL
    # =========================================================================
    print("\nLoading Kronos model...")
    sys.path.insert(0, "/root/kronos_repo")
    from model import Kronos, KronosTokenizer, KronosPredictor

    model_cache = Path(MODEL_CACHE)
    model_cache.mkdir(parents=True, exist_ok=True)

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

    vol.commit()

    # =========================================================================
    # 5. RUN BATCH PREDICTIONS
    # =========================================================================
    print(f"\nRunning predictions on {len(valid_data)} stocks...")
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)

    predictions = {}
    symbols_list = list(valid_data.keys())
    batch_size = 32

    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i+batch_size]
        for sym in batch:
            try:
                df = valid_data[sym]
                x_df = df.tail(520)
                x_hist = x_df.iloc[:512][["open", "high", "low", "close", "volume"]].copy()
                x_hist["amount"] = 0.0
                x_timestamp = x_df.iloc[:512]["timestamp"]
                y_timestamp = x_df.iloc[512:512+10]["timestamp"]

                if len(y_timestamp) < 10:
                    continue

                pred_df = predictor.predict(
                    df=x_hist,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=10,
                    T=0.8,
                    top_p=0.9,
                    sample_count=1,
                )

                current_close = float(df.iloc[-1]["close"])
                predicted_close = float(pred_df["close"].iloc[-1])
                predicted_return = (predicted_close - current_close) / current_close
                pred_std = float(pred_df["close"].std())
                confidence = abs(predicted_return) / (pred_std / current_close + 1e-8)

                predictions[sym] = {
                    "current_close": current_close,
                    "predicted_close": predicted_close,
                    "predicted_return": predicted_return,
                    "confidence": confidence,
                    "pred_std_pct": pred_std / current_close,
                }
            except Exception as e:
                continue

        if (i + batch_size) % 100 == 0:
            print(f"  Predicted {min(i+batch_size, len(symbols_list))}/{len(symbols_list)}")

    print(f"\nCompleted predictions for {len(predictions)} stocks")

    # =========================================================================
    # 6. GENERATE TRADING SIGNALS
    # =========================================================================
    signals = _generate_signals(predictions, valid_data)
    print(f"\nGenerated {len(signals)} signals")
    print(f"  Buy signals:  {len([s for s in signals if s['action'] == 'buy'])}")
    print(f"  Sell signals: {len([s for s in signals if s['action'] == 'sell'])}")

    # =========================================================================
    # 7. EXECUTE PAPER TRADES
    # =========================================================================
    print("\nExecuting trades...")
    current_prices = {s: p["current_close"] for s, p in predictions.items()}

    # Check stop-loss / take-profit
    portfolio = _Portfolio(DB_PATH, initial_cash=100_000.0)
    sl_tp_actions = portfolio.check_stop_loss_take_profit(current_prices)
    for a in sl_tp_actions:
        print(f"  {a}")

    # Rebalance
    rebalance_actions = portfolio.rebalance(
        signals, current_prices,
        max_positions=50,
        position_size_pct=0.02,
    )
    for a in rebalance_actions:
        print(f"  {a}")

    # =========================================================================
    # 8. TAKE SNAPSHOT & SEND EMAIL
    # =========================================================================
    summary = portfolio.get_summary(current_prices)
    report = _format_report(summary, signals, sl_tp_actions + rebalance_actions)

    print(f"\n{'='*60}")
    print(report)
    print(f"{'='*60}")

    # Send email
    sender = os.environ.get("KRONOS_EMAIL", "")
    password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
    recipient = os.environ.get("KRONOS_RECIPIENT", sender)

    if sender and password and recipient:
        _send_email(
            subject=f"Kronos Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body=report,
            sender_email=sender,
            sender_password=password,
            recipient_email=recipient,
        )
    else:
        print("Email not configured — skipping notification")

    vol.commit()
    print("\nTrading cycle complete.")
    return summary


# =============================================================================
# HELPER FUNCTIONS (defined at module level so they're serializable)
# =============================================================================

def _fetch_stock_universe(api_keys: list[str]) -> list[str]:
    """Fetch top US stocks from Twelve Data."""
    key_idx = 0
    symbols = []

    for exchange in ["NASDAQ", "NYSE", "AMEX"]:
        key = api_keys[key_idx % len(api_keys)]
        key_idx += 1
        try:
            resp = requests.get(
                "https://api.twelvedata.com/stocks",
                params={"exchange": exchange, "apikey": key},
                timeout=30,
            )
            data = resp.json().get("data", [])
            for s in data:
                sym = s.get("symbol", "")
                if sym and sym not in symbols:
                    symbols.append(sym)
        except Exception as e:
            print(f"Error fetching {exchange}: {e}")
        time.sleep(1)

    return symbols[:1200]


def _fetch_ohlcv_batch(
    api_keys: list[str],
    symbols: list[str],
    outputsize: int = 600,
) -> dict:
    """Fetch OHLCV data with key rotation."""
    results = {}
    key_idx = [0]
    call_times = {k: [] for k in api_keys}
    min_interval = 7.5

    def get_key():
        key = api_keys[key_idx[0] % len(api_keys)]
        key_idx[0] += 1
        now = time.time()
        call_times[key] = [t for t in call_times[key] if now - t < 60]
        if len(call_times[key]) >= 8:
            sleep_time = 60 - (now - call_times[key][0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        call_times[key].append(time.time())
        return key

    def fetch_one(sym):
        key = get_key()
        try:
            resp = requests.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": sym, "interval": "1day", "outputsize": outputsize, "apikey": key},
                timeout=30,
            )
            data = resp.json()
            if "values" not in data:
                return sym, None
            df = pd.DataFrame(data["values"])
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return sym, df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            return sym, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_one, s): s for s in symbols}
        done = 0
        for future in as_completed(futures):
            done += 1
            sym, df = future.result()
            if df is not None and len(df) >= 50:
                results[sym] = df
            if done % 100 == 0:
                print(f"  Fetched {done}/{len(symbols)} stocks...")

    return results


def _load_cached_data(cache_dir: Path) -> dict:
    """Load cached parquet files."""
    data = {}
    for f in cache_dir.glob("*.parquet"):
        try:
            data[f.stem] = pd.read_parquet(f)
        except Exception:
            pass
    return data


def _update_latest_bars(
    api_keys: list[str],
    symbols: list[str],
    cached_data: dict,
    cache_dir: Path,
    num_bars: int = 10,
):
    """Fetch only latest bars and update cache."""
    key_idx = [0]

    def get_key():
        key = api_keys[key_idx[0] % len(api_keys)]
        key_idx[0] += 1
        return key

    for i, sym in enumerate(symbols):
        key = get_key()
        try:
            resp = requests.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": sym, "interval": "1day", "outputsize": num_bars, "apikey": key},
                timeout=30,
            )
            data = resp.json()
            if "values" not in data:
                continue
            new_df = pd.DataFrame(data["values"])
            new_df = new_df.rename(columns={"datetime": "timestamp"})
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

            combined = pd.concat([cached_data[sym], new_df]).drop_duplicates(
                subset=["timestamp"], keep="last"
            ).sort_values("timestamp").reset_index(drop=True).tail(600)
            cached_data[sym] = combined
            combined.to_parquet(cache_dir / f"{sym}.parquet", index=False)
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  Updated {i+1}/{len(symbols)} stocks...")

        time.sleep(0.1)


def _generate_signals(predictions: dict, stock_data: dict) -> list[dict]:
    """Generate ranked trading signals."""
    signals = []
    for sym, pred in predictions.items():
        predicted_return = pred.get("predicted_return", 0)
        confidence = pred.get("confidence", 0)
        pred_std_pct = pred.get("pred_std_pct", 0)

        if confidence < 0.005 or pred_std_pct > 0.08:
            continue

        vol_score = 1.0 / (pred_std_pct + 0.01)
        score = 0.60 * predicted_return + 0.25 * np.tanh(confidence) + 0.15 * np.tanh(vol_score)

        hist = stock_data.get(sym)
        if hist is not None and len(hist) >= 20:
            closes = hist["close"].values
            sma20 = np.mean(closes[-20:])
            sma5 = np.mean(closes[-5:])
            momentum = (sma5 - sma20) / sma20
            if momentum > 0:
                score *= 1.2

        action = "buy" if predicted_return > 0.005 else ("sell" if predicted_return < -0.005 else "hold")
        signals.append({
            "symbol": sym, "action": action, "score": score,
            "predicted_return": predicted_return, "confidence": confidence,
            "current_close": pred["current_close"],
            "predicted_close": pred.get("predicted_close", 0),
        })

    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals


class _Portfolio:
    """Inline portfolio tracker."""

    def __init__(self, db_path: str, initial_cash: float = 100_000.0):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initial_cash = initial_cash
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY, shares REAL DEFAULT 0,
                    avg_cost REAL DEFAULT 0, entry_date TEXT,
                    stop_loss REAL, take_profit REAL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT,
                    action TEXT, shares REAL, price REAL, total REAL,
                    timestamp TEXT, reason TEXT
                );
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT UNIQUE,
                    cash REAL, equity REAL, total_value REAL,
                    positions_json TEXT, daily_pnl REAL, cumulative_pnl REAL
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
            """)
            existing = conn.execute("SELECT value FROM state WHERE key='cash'").fetchone()
            if not existing:
                conn.execute("INSERT INTO state (key, value) VALUES ('cash', ?)", (str(self.initial_cash),))

    def get_cash(self) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM state WHERE key='cash'").fetchone()
            return float(row[0]) if row else self.initial_cash

    def get_positions(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return {r["symbol"]: dict(r) for r in conn.execute("SELECT * FROM positions WHERE shares > 0").fetchall()}

    def get_total_equity(self, prices: dict) -> float:
        return sum(pos["shares"] * prices.get(sym, pos["avg_cost"]) for sym, pos in self.get_positions().items())

    def get_total_value(self, prices: dict) -> float:
        return self.get_cash() + self.get_total_equity(prices)

    def buy(self, symbol, price, shares, reason="signal"):
        cost = price * shares
        cash = self.get_cash()
        if cost > cash:
            shares = int(cash / price)
            if shares <= 0:
                return False
            cost = price * shares
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute("SELECT shares, avg_cost FROM positions WHERE symbol=?", (symbol,)).fetchone()
            if existing:
                old_sh, old_avg = existing
                new_sh = old_sh + shares
                new_avg = (old_sh * old_avg + cost) / new_sh
                conn.execute("UPDATE positions SET shares=?, avg_cost=? WHERE symbol=?", (new_sh, new_avg, symbol))
            else:
                conn.execute("INSERT INTO positions (symbol, shares, avg_cost, entry_date, stop_loss, take_profit) VALUES (?,?,?,?,?,?)",
                    (symbol, shares, price, datetime.now().isoformat(), price*0.95, price*1.15))
            conn.execute("INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?,?,?,?,?,?,?)",
                (symbol, "buy", shares, price, cost, datetime.now().isoformat(), reason))
            conn.execute("UPDATE state SET value=? WHERE key='cash'", (str(cash - cost),))
        return True

    def sell(self, symbol, price, reason="signal"):
        with sqlite3.connect(self.db_path) as conn:
            pos = conn.execute("SELECT shares, avg_cost FROM positions WHERE symbol=? AND shares>0", (symbol,)).fetchone()
            if not pos:
                return False
            sh, avg = pos
            pnl = (price - avg) * sh
            conn.execute("UPDATE positions SET shares=0 WHERE symbol=?", (symbol,))
            conn.execute("INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?,?,?,?,?,?,?)",
                (symbol, "sell", sh, price, price*sh, datetime.now().isoformat(), reason))
            cash = self.get_cash()
            conn.execute("UPDATE state SET value=? WHERE key='cash'", (str(cash + price*sh),))
        return True

    def check_stop_loss_take_profit(self, prices):
        actions = []
        for sym, pos in self.get_positions().items():
            p = prices.get(sym)
            if p and p <= pos["stop_loss"]:
                self.sell(sym, p, "stop_loss")
                actions.append(f"STOP LOSS: {sym} @ ${p:.2f}")
            elif p and p >= pos["take_profit"]:
                self.sell(sym, p, "take_profit")
                actions.append(f"TAKE PROFIT: {sym} @ ${p:.2f}")
        return actions

    def rebalance(self, signals, prices, max_positions=50, position_size_pct=0.02):
        actions = []
        total_val = self.get_total_value(prices)
        cash = self.get_cash()
        buy_signals = [s for s in signals if s["action"] == "buy"][:max_positions]
        target = {s["symbol"] for s in buy_signals}

        for sym, pos in self.get_positions().items():
            if sym not in target:
                p = prices.get(sym, pos["avg_cost"])
                self.sell(sym, p, "rebalance_out")
                actions.append(f"SELL: {sym}")

        pos_size = total_val * position_size_pct
        for sig in buy_signals:
            sym = sig["symbol"]
            p = prices.get(sym, sig["current_close"])
            cur = self.get_positions().get(sym, {}).get("shares", 0)
            if cur > 0:
                continue
            if cash < pos_size:
                break
            shares = int(pos_size / p)
            if shares > 0:
                self.buy(sym, p, shares, "rebalance_in")
                actions.append(f"BUY: {sym} x{shares}")
                cash = self.get_cash()
        return actions

    def get_summary(self, prices):
        cash = self.get_cash()
        equity = self.get_total_equity(prices)
        total = cash + equity
        positions = self.get_positions()
        with sqlite3.connect(self.db_path) as conn:
            prev = conn.execute("SELECT total_value FROM portfolio_snapshots ORDER BY date DESC LIMIT 1").fetchone()
            prev_total = float(prev[0]) if prev else self.initial_cash
            today = datetime.now().strftime("%Y-%m-%d")
            daily_pnl = total - prev_total
            cum_pnl = total - self.initial_cash
            conn.execute("INSERT OR REPLACE INTO portfolio_snapshots (date,cash,equity,total_value,positions_json,daily_pnl,cumulative_pnl) VALUES (?,?,?,?,?,?,?)",
                (today, cash, equity, total, json.dumps({s:dict(p) for s,p in positions.items()}), daily_pnl, cum_pnl))
            trades = conn.execute("SELECT symbol,action,shares,price,total,timestamp,reason FROM trades ORDER BY timestamp DESC LIMIT 10").fetchall()
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

        pos_pnl = []
        for sym, pos in positions.items():
            if pos["shares"] <= 0:
                continue
            p = prices.get(sym, pos["avg_cost"])
            pnl = (p - pos["avg_cost"]) * pos["shares"]
            pos_pnl.append({"symbol": sym, "shares": pos["shares"], "avg_cost": pos["avg_cost"],
                "current_price": p, "pnl": pnl, "pnl_pct": (p-pos["avg_cost"])/pos["avg_cost"]*100})
        pos_pnl.sort(key=lambda x: x["pnl"], reverse=True)

        return {"date": today, "cash": cash, "equity": equity, "total_value": total,
            "daily_pnl": daily_pnl, "cumulative_pnl": cum_pnl,
            "num_positions": len([p for p in positions.values() if p["shares"]>0]),
            "top_winners": pos_pnl[:5], "top_losers": pos_pnl[-5:],
            "total_trades": total_trades, "recent_trades": trades}


def _format_report(summary, signals, actions):
    lines = [
        "="*60, "  KRONOS TRADING ENGINE — DAILY REPORT", f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}", "="*60, "",
        "PORTFOLIO SUMMARY", "-"*40,
        f"  Cash:           ${summary['cash']:>12,.2f}",
        f"  Equity:         ${summary['equity']:>12,.2f}",
        f"  Total Value:    ${summary['total_value']:>12,.2f}",
        f"  Daily P&L:      ${summary['daily_pnl']:>12,.2f}",
        f"  Cumulative P&L: ${summary['cumulative_pnl']:>12,.2f}",
        f"  Positions:      {summary['num_positions']}",
        f"  Total Trades:   {summary['total_trades']}", "",
    ]
    if actions:
        lines.append("TODAY'S ACTIONS")
        lines.append("-"*40)
        for a in actions:
            lines.append(f"  {a}")
        lines.append("")
    if summary.get("top_winners"):
        lines.append("TOP WINNERS")
        lines.append("-"*40)
        for p in summary["top_winners"]:
            lines.append(f"  {p['symbol']:>6s}  {p['shares']:.0f} sh  ${p['current_price']:>8.2f}  P&L: ${p['pnl']:>+8.2f} ({p['pnl_pct']:>+.1f}%)")
        lines.append("")
    if summary.get("top_losers"):
        lines.append("TOP LOSERS")
        lines.append("-"*40)
        for p in summary["top_losers"]:
            lines.append(f"  {p['symbol']:>6s}  {p['shares']:.0f} sh  ${p['current_price']:>8.2f}  P&L: ${p['pnl']:>+8.2f} ({p['pnl_pct']:>+.1f}%)")
        lines.append("")
    if summary.get("recent_trades"):
        lines.append("RECENT TRADES")
        lines.append("-"*40)
        for t in summary["recent_trades"][:10]:
            lines.append(f"  {t[5][:16]}  {t[1].upper():>4s}  {t[0]:>6s}  {t[2]:.0f} x ${t[3]:.2f} = ${t[4]:.2f}  ({t[6]})")
        lines.append("")
    lines.extend(["="*60, "  Generated by Kronos Trading Engine", "="*60])
    return "\n".join(lines)


def _send_email(subject, body, sender_email, sender_password, recipient_email):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(f"<pre>{body}</pre>", "html"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Email error: {e}")


# =============================================================================
# CRON SCHEDULES
# =============================================================================
@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    schedule=modal.Cron("30 9 * * 1-5"),  # 9:30 AM ET, weekdays
)
def pre_market_run():
    """Pre-market data fetch + prediction (runs before market open)."""
    run_trading_cycle.local()


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    schedule=modal.Cron("45 15 * * 1-5"),  # 3:45 PM ET, weekdays
)
def post_market_run():
    """Post-market update + rebalance (runs before market close)."""
    run_trading_cycle.local()
