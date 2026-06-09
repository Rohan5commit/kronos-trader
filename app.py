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

# Add strategy module to path
sys.path.insert(0, "/root/app")
from strategy.portfolio import Portfolio as _Portfolio

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

project_dir = Path(__file__).parent

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
    mounts=[modal.Mount.from_local_dir(project_dir, remote_path="/root/app")],
    gpu="T4",
    timeout=1800,
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
    symbols_to_update = [s for s in all_symbols if s in cached_data]

    print(f"\nData status: {len(cached_data)} cached, {len(symbols_to_fetch)} new, {len(symbols_to_update)} to update")

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

    if symbols_to_update:
        print(f"\nUpdating {len(symbols_to_update)} stocks with latest bars...")
        _update_latest_bars(API_KEYS, symbols_to_update, cached_data, data_cache_dir, requests, pd)
        vol.commit()

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

    vol.commit()
    print("\nTrading cycle complete.")
    return summary


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
    """Fetch only latest bars and update cache — parallelized."""
    key_idx = [0]
    lock = __import__('threading').Lock()
    def get_key():
        with lock:
            key = api_keys[key_idx[0] % len(api_keys)]
            key_idx[0] += 1
            return key

    updated = [0]
    def _update_one(sym):
        key = get_key()
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
    with ThreadPoolExecutor(max_workers=32) as executor:
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
# CRON SCHEDULES
# =============================================================================
@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    schedule=modal.Cron("30 9 * * 1-5"),
)
def pre_market_run():
    """Pre-market: fetch data + predict + trade, 9:30 AM ET weekdays."""
    run_trading_cycle.local()


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    schedule=modal.Cron("45 15 * * 1-5"),
)
def post_market_run():
    """Post-market: update data + rebalance + email report, 3:45 PM ET weekdays."""
    run_trading_cycle.local(send_email=True)
