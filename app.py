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
import modal
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =============================================================================
# Modal App Setup
# =============================================================================
app = modal.App("kronos-trader")

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

vol = modal.Volume.from_name("kronos-data", create_if_missing=True)

VOLUME_PATH = "/kronos-data"
MODEL_CACHE = f"{VOLUME_PATH}/model"
DATA_CACHE = f"{VOLUME_PATH}/ohlcv"
DB_PATH = f"{VOLUME_PATH}/trades.db"
CONTEXT_PATH = f"{VOLUME_PATH}/context.json"


# =============================================================================
# Core Trading Function
# =============================================================================
@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    gpu="T4",
    timeout=3600,
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
        modal.Secret.from_name("kronos-email"),
    ],
    memory=16384,
)
def run_trading_cycle(send_email=False):
    """Main trading cycle — fetches data, runs inference, trades, emails report."""
    import torch
    import numpy as np
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n{'='*60}")
    print(f"  KRONOS TRADING ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

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
    # 1b. READ PREVIOUS CONTEXT
    # =========================================================================
    prev_context = _read_context()
    if prev_context:
        print(f"Loaded previous run context from {prev_context.get('run_timestamp', '?')}")
        prev_actions = prev_context.get("actions_this_run", {})
        print(f"  Previous run: {prev_actions.get('total_buys', 0)} buys, {prev_actions.get('total_sells', 0)} sells")
        prev_port = prev_context.get("portfolio", {})
        print(f"  Previous portfolio: ${prev_port.get('total_value', 0):,.2f} ({prev_port.get('num_positions', 0)} positions)")
    else:
        print("No previous context found (first run)")

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
        all_symbols = _fetch_stock_universe(API_KEYS, requests)
        stock_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stock_list_path, "w") as f:
            json.dump(all_symbols, f)
        print(f"Cached {len(all_symbols)} symbols")

    # =========================================================================
    # 3. FETCH OHLCV DATA (incremental)
    # =========================================================================
    data_cache_dir = Path(DATA_CACHE)
    data_cache_dir.mkdir(parents=True, exist_ok=True)

    cached_data = _load_cached_data(data_cache_dir, pd)
    symbols_to_fetch = [s for s in all_symbols if s not in cached_data]

    print(f"\nData status: {len(cached_data)} cached, {len(symbols_to_fetch)} new")

    if symbols_to_fetch:
        print(f"\nFetching {len(symbols_to_fetch)} new stocks in batches of 200...")
        for batch_start in range(0, len(symbols_to_fetch), 200):
            batch = symbols_to_fetch[batch_start:batch_start+200]
            print(f"  Batch {batch_start//200+1}: fetching {len(batch)} stocks...")
            new_data = _fetch_ohlcv_batch(API_KEYS, batch, requests, pd, outputsize=520)
            for sym, df in new_data.items():
                df.to_parquet(data_cache_dir / f"{sym}.parquet", index=False)
                cached_data[sym] = df
            vol.commit()
            print(f"  Committed {len(new_data)} stocks to volume")

    valid_data = {
        s: df for s, df in cached_data.items()
        if len(df) >= 100
    }
    print(f"\nValid stocks for prediction: {len(valid_data)}")

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

    LOOKBACK = 400  # Must be < max_context (512) per Kronos design
    PRED_LEN = 10

    predictions = {}
    symbols_list = list(valid_data.keys())
    batch_size = 32

    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i+batch_size]
        for sym in batch:
            try:
                df = valid_data[sym]
                if len(df) < LOOKBACK + 10:
                    continue

                x_hist = df.iloc[:LOOKBACK][["open", "high", "low", "close", "volume"]].copy()
                x_hist["amount"] = 0.0
                x_timestamp = pd.Series(df.iloc[:LOOKBACK]["timestamp"])

                # Generate synthetic future timestamps (business day freq from last timestamp)
                last_ts = x_timestamp.iloc[-1]
                y_timestamp = pd.Series(pd.date_range(
                    start=last_ts + pd.Timedelta(days=1),
                    periods=PRED_LEN,
                    freq="B",
                ))

                # Validate no NaN
                if x_hist.isnull().values.any():
                    continue

                pred_df = predictor.predict(
                    df=x_hist,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=PRED_LEN,
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
                if len(predictions) <= 5:
                    print(f"  ✓ {sym}: return={predicted_return:+.2%}, conf={confidence:.3f}")
            except Exception as e:
                if len(predictions) < 3:
                    print(f"  ✗ {sym}: {type(e).__name__}: {e}")
                continue

        if (i + batch_size) % 100 == 0:
            print(f"  Predicted {min(i+batch_size, len(symbols_list))}/{len(symbols_list)}")

    print(f"\nCompleted predictions for {len(predictions)} stocks")

    # =========================================================================
    # 6. GENERATE TRADING SIGNALS
    # =========================================================================
    signals = _generate_signals(predictions, valid_data, np)
    print(f"\nGenerated {len(signals)} signals")
    print(f"  Buy signals:  {len([s for s in signals if s['action'] == 'buy'])}")
    print(f"  Sell signals: {len([s for s in signals if s['action'] == 'sell'])}")

    # =========================================================================
    # 7. EXECUTE PAPER TRADES (model-driven)
    # =========================================================================
    print("\nExecuting trades (model-driven)...")
    current_prices = {s: p["current_close"] for s, p in predictions.items()}

    portfolio = _Portfolio(DB_PATH, initial_cash=100_000.0)
    trade_actions = portfolio.manage_positions(
        predictions, current_prices,
        max_positions=50,
        min_confidence=0.005,
    )

    # =========================================================================
    # 8. TAKE SNAPSHOT & SEND EMAIL
    # =========================================================================
    summary = portfolio.get_summary(current_prices)
    report = _format_report(summary, signals, trade_actions)

    print(f"\n{'='*60}")
    print(report)
    print(f"{'='*60}")

    sender = os.environ.get("KRONOS_EMAIL", "")
    password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
    recipient = os.environ.get("KRONOS_RECIPIENT", sender)

    if send_email and sender and password and recipient:
        _send_email(
            subject=f"Kronos Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body=report,
            sender_email=sender,
            sender_password=password,
            recipient_email=recipient,
        )
    else:
        print("Email not configured — skipping notification")

    # =========================================================================
    # 9. WRITE CONTEXT FOR NEXT RUN
    # =========================================================================
    _write_context(trade_actions, predictions, signals, summary, prev_context)

    vol.commit()
    print("\nTrading cycle complete.")
    return summary


# =============================================================================
# MODEL-DRIVEN PORTFOLIO
# =============================================================================
class _Portfolio:
    """Fully model-driven portfolio. No hard-coded stop-loss/take-profit."""

    def __init__(self, db_path, initial_cash=100_000.0):
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
                    last_prediction REAL DEFAULT 0
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
            # Migration: add last_prediction column if missing
            cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
            if "last_prediction" not in cols:
                conn.execute("ALTER TABLE positions ADD COLUMN last_prediction REAL DEFAULT 0")
            existing = conn.execute("SELECT value FROM state WHERE key='cash'").fetchone()
            if not existing:
                conn.execute("INSERT INTO state (key, value) VALUES ('cash', ?)", (str(self.initial_cash),))

    def get_cash(self):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM state WHERE key='cash'").fetchone()
            return float(row[0]) if row else self.initial_cash

    def get_positions(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return {r["symbol"]: dict(r) for r in conn.execute("SELECT * FROM positions WHERE shares > 0").fetchall()}

    def get_total_equity(self, prices):
        return sum(pos["shares"] * prices.get(sym, pos["avg_cost"]) for sym, pos in self.get_positions().items())

    def get_total_value(self, prices):
        return self.get_cash() + self.get_total_equity(prices)

    def buy(self, symbol, price, shares, reason="signal", predicted_return=0.0):
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
                conn.execute(
                    "INSERT INTO positions (symbol, shares, avg_cost, entry_date, last_prediction) VALUES (?,?,?,?,?)",
                    (symbol, shares, price, datetime.now().isoformat(), predicted_return),
                )
            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?,?,?,?,?,?,?)",
                (symbol, "buy", shares, price, cost, datetime.now().isoformat(), reason),
            )
            conn.execute("UPDATE state SET value=? WHERE key='cash'", (str(cash - cost),))
        print(f"  BUY  {shares:>6} x {symbol:<8} @ ${price:>8.2f}  (${cost:>10.2f})  pred={predicted_return:+.2%}")
        return True

    def sell(self, symbol, price, reason="signal"):
        with sqlite3.connect(self.db_path) as conn:
            pos = conn.execute("SELECT shares, avg_cost FROM positions WHERE symbol=? AND shares>0", (symbol,)).fetchone()
            if not pos:
                return False
            sh, avg = pos
            pnl = (price - avg) * sh
            pnl_pct = (price - avg) / avg * 100 if avg else 0
            conn.execute("UPDATE positions SET shares=0 WHERE symbol=?", (symbol,))
            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?,?,?,?,?,?,?)",
                (symbol, "sell", sh, price, price * sh, datetime.now().isoformat(), reason),
            )
            cash = self.get_cash()
            conn.execute("UPDATE state SET value=? WHERE key='cash'", (str(cash + price * sh),))
        print(f"  SELL {sh:>6} x {symbol:<8} @ ${price:>8.2f}  (P&L: ${pnl:>+10.2f} {pnl_pct:>+6.1f}%)  {reason}")
        return True

    def manage_positions(self, predictions, current_prices, max_positions=50, min_confidence=0.005):
        """Model-driven: sell where model predicts negative, buy top predictions."""
        actions = []
        positions = self.get_positions()

        # SELL: model predicts negative return or no prediction available
        for sym, pos in list(positions.items()):
            price = current_prices.get(sym)
            if price is None:
                continue
            pred = predictions.get(sym)
            if pred is None:
                self.sell(sym, price, "no_prediction")
                actions.append(f"SELL {sym} (no prediction)")
                continue
            ret = pred.get("predicted_return", 0)
            conf = pred.get("confidence", 0)
            if ret < 0 or conf < min_confidence:
                self.sell(sym, price, f"model_sell ret={ret:+.2%} conf={conf:.3f}")
                actions.append(f"SELL {sym} (ret={ret:+.2%})")

        # BUY: rank by return * confidence, position size scaled by confidence
        positions = self.get_positions()
        total_value = self.get_total_value(current_prices)
        cash = self.get_cash()
        current_count = len([p for p in positions.values() if p["shares"] > 0])
        slots = max_positions - current_count

        candidates = []
        for sym, pred in predictions.items():
            if sym in positions and positions[sym].get("shares", 0) > 0:
                continue
            ret = pred.get("predicted_return", 0)
            conf = pred.get("confidence", 0)
            if ret <= 0 or conf < min_confidence:
                continue
            candidates.append((sym, pred, ret * conf))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for sym, pred, score in candidates[:slots]:
            if cash < 100:
                break
            price = current_prices.get(sym, pred.get("current_close", 0))
            if price <= 0:
                continue
            conf = pred.get("confidence", 0)
            ret = pred.get("predicted_return", 0)
            conf_mult = min(conf / 0.05, 1.0)
            position_pct = 0.01 + 0.04 * conf_mult
            position_value = min(total_value * position_pct, cash * 0.95)
            shares = int(position_value / price)
            if shares > 0:
                if self.buy(sym, price, shares, "model_buy", predicted_return=ret):
                    actions.append(f"BUY {sym} x{shares} (ret={ret:+.2%}, conf={conf:.3f})")
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
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_snapshots (date,cash,equity,total_value,positions_json,daily_pnl,cumulative_pnl) VALUES (?,?,?,?,?,?,?)",
                (today, cash, equity, total, json.dumps({s: dict(p) for s, p in positions.items()}), daily_pnl, cum_pnl),
            )
            trades = conn.execute(
                "SELECT symbol,action,shares,price,total,timestamp,reason FROM trades ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()
            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

        pos_pnl = []
        for sym, pos in positions.items():
            if pos["shares"] <= 0:
                continue
            p = prices.get(sym, pos["avg_cost"])
            pnl = (p - pos["avg_cost"]) * pos["shares"]
            pos_pnl.append({
                "symbol": sym, "shares": pos["shares"], "avg_cost": pos["avg_cost"],
                "current_price": p, "pnl": pnl,
                "pnl_pct": (p - pos["avg_cost"]) / pos["avg_cost"] * 100 if pos["avg_cost"] else 0,
            })
        pos_pnl.sort(key=lambda x: x["pnl"], reverse=True)
        return {
            "date": today, "cash": cash, "equity": equity, "total_value": total,
            "daily_pnl": daily_pnl, "cumulative_pnl": cum_pnl,
            "num_positions": len([p for p in positions.values() if p["shares"] > 0]),
            "top_winners": pos_pnl[:5], "top_losers": pos_pnl[-5:],
            "total_trades": total_trades, "recent_trades": trades,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _fetch_stock_universe(api_keys, requests_mod):
    """Fetch top US stocks from Twelve Data."""
    key_idx = 0
    symbols = []
    for exchange in ["NASDAQ", "NYSE", "AMEX"]:
        key = api_keys[key_idx % len(api_keys)]
        key_idx += 1
        try:
            resp = requests_mod.get(
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


def _fetch_ohlcv_batch(api_keys, symbols, requests_mod, pd_mod, outputsize=600):
    """Fetch OHLCV data with key rotation."""
    results = {}
    key_idx = [0]
    call_times = {k: [] for k in api_keys}

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
            resp = requests_mod.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": sym, "interval": "1day", "outputsize": outputsize, "apikey": key},
                timeout=30,
            )
            data = resp.json()
            if "values" not in data:
                return sym, None
            df = pd_mod.DataFrame(data["values"])
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd_mod.to_datetime(df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd_mod.to_numeric(df[col], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return sym, df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            return sym, None

    from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _load_cached_data(cache_dir, pd_mod):
    """Load cached parquet files."""
    data = {}
    for f in cache_dir.glob("*.parquet"):
        try:
            data[f.stem] = pd_mod.read_parquet(f)
        except Exception:
            pass
    return data


def _update_latest_bars(api_keys, symbols, cached_data, cache_dir, requests_mod, pd_mod, num_bars=10):
    """Fetch only latest bars and update cache — rate-limited to respect API limits."""
    import threading
    import time as _time

    # Rate limit: 8 calls/min/key, 8 keys = 64 calls/min total
    # Use 8 workers with a per-key cooldown of 8s (60/8 = 7.5s per key cycle)
    key_last_used = {k: 0.0 for k in api_keys}
    lock = threading.Lock()

    def get_key_throttled():
        while True:
            with lock:
                now = _time.time()
                best_key = None
                best_wait = float('inf')
                for k in api_keys:
                    wait = max(0, 7.5 - (now - key_last_used[k]))
                    if wait < best_wait:
                        best_wait = wait
                        best_key = k
                if best_wait <= 0.1:
                    key_last_used[best_key] = now
                    return best_key
            _time.sleep(0.5)

    updated = [0]
    def _update_one(sym):
        key = get_key_throttled()
        try:
            resp = requests_mod.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": sym, "interval": "1day", "outputsize": num_bars, "apikey": key},
                timeout=30,
            )
            data = resp.json()
            if "values" not in data:
                return
            new_df = pd_mod.DataFrame(data["values"])
            new_df = new_df.rename(columns={"datetime": "timestamp"})
            new_df["timestamp"] = pd_mod.to_datetime(new_df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                new_df[col] = pd_mod.to_numeric(new_df[col], errors="coerce")
            combined = pd_mod.concat([cached_data[sym], new_df]).drop_duplicates(
                subset=["timestamp"], keep="last"
            ).sort_values("timestamp").reset_index(drop=True).tail(600)
            cached_data[sym] = combined
            combined.to_parquet(cache_dir / f"{sym}.parquet", index=False)
        except Exception:
            pass
        with lock:
            updated[0] += 1
            if updated[0] % 200 == 0:
                print(f"  Updated {updated[0]}/{len(symbols)} stocks...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_update_one, s): s for s in symbols}
        for f in as_completed(futures):
            f.result()

    print(f"  Updated {updated[0]}/{len(symbols)} stocks total")


def _generate_signals(predictions, stock_data, np_mod):
    """Generate ranked trading signals."""
    signals = []
    for sym, pred in predictions.items():
        predicted_return = pred.get("predicted_return", 0)
        confidence = pred.get("confidence", 0)
        pred_std_pct = pred.get("pred_std_pct", 0)

        if confidence < 0.005 or pred_std_pct > 0.08:
            continue

        vol_score = 1.0 / (pred_std_pct + 0.01)
        score = 0.60 * predicted_return + 0.25 * float(np_mod.tanh(confidence)) + 0.15 * float(np_mod.tanh(vol_score))

        hist = stock_data.get(sym)
        if hist is not None and len(hist) >= 20:
            closes = hist["close"].values
            sma20 = float(np_mod.mean(closes[-20:]))
            sma5 = float(np_mod.mean(closes[-5:]))
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


def _format_report(summary, signals, actions):
    lines = [
        "=" * 60,
        "  KRONOS TRADING ENGINE — DAILY REPORT",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "PORTFOLIO SUMMARY",
        "-" * 40,
        f"  Cash:           ${summary['cash']:>12,.2f}",
        f"  Equity:         ${summary['equity']:>12,.2f}",
        f"  Total Value:    ${summary['total_value']:>12,.2f}",
        f"  Daily P&L:      ${summary['daily_pnl']:>12,.2f}",
        f"  Cumulative P&L: ${summary['cumulative_pnl']:>12,.2f}",
        f"  Positions:      {summary['num_positions']}",
        f"  Total Trades:   {summary['total_trades']}",
        "",
    ]
    if actions:
        lines.append("TODAY'S ACTIONS")
        lines.append("-" * 40)
        for a in actions:
            lines.append(f"  {a}")
        lines.append("")
    if summary.get("top_winners"):
        lines.append("TOP WINNERS")
        lines.append("-" * 40)
        for p in summary["top_winners"]:
            lines.append(
                f"  {p['symbol']:>6s}  {p['shares']:.0f} sh  "
                f"${p['current_price']:>8.2f}  P&L: ${p['pnl']:>+8.2f} ({p['pnl_pct']:>+.1f}%)"
            )
        lines.append("")
    if summary.get("top_losers"):
        lines.append("TOP LOSERS")
        lines.append("-" * 40)
        for p in summary["top_losers"]:
            lines.append(
                f"  {p['symbol']:>6s}  {p['shares']:.0f} sh  "
                f"${p['current_price']:>8.2f}  P&L: ${p['pnl']:>+8.2f} ({p['pnl_pct']:>+.1f}%)"
            )
        lines.append("")
    if summary.get("recent_trades"):
        lines.append("RECENT TRADES")
        lines.append("-" * 40)
        for t in summary["recent_trades"][:10]:
            lines.append(
                f"  {t[5][:16]}  {t[1].upper():>4s}  {t[0]:>6s}  "
                f"{t[2]:.0f} x ${t[3]:.2f} = ${t[4]:.2f}  ({t[6]})"
            )
        lines.append("")
    lines.extend(["=" * 60, "  Generated by Kronos Trading Engine", "=" * 60])
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
# CONTEXT (run-to-run memory)
# =============================================================================
def _read_context():
    """Read previous run's context file. Returns dict or None."""
    ctx_path = Path(CONTEXT_PATH)
    if not ctx_path.exists():
        return None
    try:
        with open(ctx_path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_context(trade_actions, predictions, signals, summary, prev_context):
    """Write context file for the next run. Overwrites previous."""
    from datetime import datetime as _dt

    # Summarize what happened this run
    buys = [a for a in trade_actions if a.startswith("BUY")]
    sells = [a for a in trade_actions if a.startswith("SELL")]

    # Top predictions (why we bought what we bought)
    top_buys = sorted(
        [(s, p) for s, p in predictions.items() if p.get("predicted_return", 0) > 0],
        key=lambda x: x[1].get("predicted_return", 0) * x[1].get("confidence", 0),
        reverse=True,
    )[:10]

    # Top sells (why we sold what we sold)
    top_sells = [a for a in trade_actions if a.startswith("SELL")][:10]

    context = {
        "run_timestamp": _dt.now().isoformat(),
        "portfolio": {
            "cash": summary.get("cash", 0),
            "total_value": summary.get("total_value", 0),
            "cumulative_pnl": summary.get("cumulative_pnl", 0),
            "num_positions": summary.get("num_positions", 0),
        },
        "actions_this_run": {
            "buys": buys,
            "sells": sells,
            "total_buys": len(buys),
            "total_sells": len(sells),
        },
        "reasoning": {
            "top_predictions": [
                {
                    "symbol": s,
                    "return": round(p.get("predicted_return", 0), 4),
                    "confidence": round(p.get("confidence", 0), 4),
                }
                for s, p in top_buys
            ],
            "sell_reasons": top_sells,
        },
        "previous_context": prev_context,
    }

    ctx_path = Path(CONTEXT_PATH)
    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ctx_path, "w") as f:
        json.dump(context, f, indent=2)
    print(f"Context written for next run")


# =============================================================================
# CRON SCHEDULES
# =============================================================================
@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
        modal.Secret.from_name("kronos-email"),
    ],
    schedule=modal.Cron("30 9 * * 1-5"),
    timeout=3600,
    gpu="T4",
)
def pre_market_run():
    """Pre-market: predict + trade, 9:30 AM ET weekdays."""
    run_trading_cycle.local()


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
        modal.Secret.from_name("kronos-email"),
    ],
    schedule=modal.Cron("45 15 * * 1-5"),
    timeout=3600,
    gpu="T4",
)
def post_market_run():
    """Post-market: predict + rebalance + email report, 3:45 PM ET weekdays."""
    run_trading_cycle.local(send_email=True)


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
    ],
    schedule=modal.Cron("0 8 * * 1-5"),
    timeout=3600,
)
def update_data_morning():
    """Update latest bars for all cached stocks. Runs daily 8:00 AM ET before pre-market."""
    _run_data_update()


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
    ],
    schedule=modal.Cron("0 15 * * 1-5"),
    timeout=3600,
)
def update_data_afternoon():
    """Update latest bars for all cached stocks. Runs daily 3:00 PM ET before post-market."""
    _run_data_update()


def _run_data_update():
    """Shared data update logic."""
    import pandas as pd
    import requests as req

    print(f"\n{'='*60}")
    print(f"  DATA UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    keys_raw = os.environ.get("TWELVE_DATA_KEYS", "[]")
    API_KEYS = json.loads(keys_raw) if keys_raw else []
    if not API_KEYS:
        print("ERROR: No API keys found")
        return
    print(f"Loaded {len(API_KEYS)} API keys")

    stock_list_path = Path(VOLUME_PATH) / "stocks.json"
    if not stock_list_path.exists():
        print("ERROR: No stocks.json — run trading cycle first")
        return
    with open(stock_list_path) as f:
        all_symbols = json.load(f)
    print(f"Loaded {len(all_symbols)} symbols")

    data_cache_dir = Path(DATA_CACHE)
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_data = _load_cached_data(data_cache_dir, pd)

    symbols_to_update = [s for s in all_symbols if s in cached_data]
    print(f"Updating {len(symbols_to_update)} stocks...")

    _update_latest_bars(API_KEYS, symbols_to_update, cached_data, data_cache_dir, req, pd)
    vol.commit()
    print(f"\nData update complete. Updated {len(symbols_to_update)} stocks.")
