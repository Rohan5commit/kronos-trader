"""
Kronos Paper Trading Engine — Modal Serverless
Runs 2x/day on T4 GPU, fetches 1000 US stocks, predicts with Kronos, trades via Alpaca.
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
        "alpaca-py",
        "pytz",
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
        modal.Secret.from_name("kronos-alpaca"),
    ],
    memory=16384,
)
def run_trading_cycle(send_email=False):
    """Main trading cycle — fetches data, runs inference, trades via Alpaca, emails report."""
    import torch
    import numpy as np
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n{'='*60}")
    print(f"  KRONOS TRADING ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        _run_cycle_body(send_email)
    except Exception as e:
        print(f"\nFATAL ERROR: {type(e).__name__}: {e}")
        print("Attempting to send error report email...")
        if send_email:
            try:
                sender = os.environ.get("KRONOS_EMAIL", "")
                password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
                recipient = os.environ.get("KRONOS_RECIPIENT", sender)
                if sender and password and recipient:
                    _send_email(
                        subject=f"Kronos ERROR — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        body=f"FATAL ERROR in trading cycle:\n\n{type(e).__name__}: {e}\n\nCheck Modal logs for details.",
                        sender_email=sender,
                        sender_password=password,
                        recipient_email=recipient,
                    )
            except Exception:
                pass
        raise


def _run_cycle_body(send_email):
    """Inner trading cycle — separated so outer wrapper can catch all errors."""
    import torch
    import numpy as np
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # Data freshness check — warn if cached data is stale
    if cached_data:
        newest_ts = None
        for df in cached_data.values():
            if len(df) > 0:
                last = df["timestamp"].iloc[-1]
                if newest_ts is None or last > newest_ts:
                    newest_ts = last
        if newest_ts is not None:
            from datetime import timedelta
            age_days = (datetime.now() - newest_ts.to_pydatetime().replace(tzinfo=None)).days
            if age_days > 3:
                print(f"WARNING: Cached data is {age_days} days old (newest: {newest_ts.date()}). Data update cron may have failed.")
            else:
                print(f"Data freshness: OK (newest bar: {newest_ts.date()}, {age_days} days old)")

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
    if "/root/kronos_repo" not in sys.path:
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

                # Skip penny stocks BEFORE expensive GPU inference
                current_close = float(df.iloc[-1]["close"])
                if current_close < 1.0:
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

                # Cap predictions at ±20%
                predicted_return = max(-0.20, min(0.20, predicted_return))

                pred_std = float(pred_df["close"].std())
                # Confidence: ratio of predicted return magnitude to trajectory volatility
                # Higher = more certain the prediction is meaningful vs noise
                pred_range = float(pred_df["close"].max() - pred_df["close"].min())
                trajectory_stability = pred_range / (pred_std + 1e-8)
                confidence = abs(predicted_return) * min(trajectory_stability, 10.0)

                # Skip if prediction is too noisy
                pred_std_pct = pred_std / current_close
                if pred_std_pct > 0.15:
                    continue

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

        predicted_count = len(predictions)
        if predicted_count > 0 and predicted_count % 100 < batch_size:
            print(f"  Predicted {predicted_count}/{len(symbols_list)} stocks")

    print(f"\nCompleted predictions for {len(predictions)} stocks")

    # =========================================================================
    # 6. GENERATE TRADING SIGNALS
    # =========================================================================
    signals = _generate_signals(predictions, valid_data, np)
    print(f"\nGenerated {len(signals)} signals")
    print(f"  Buy signals:  {len([s for s in signals if s['action'] == 'buy'])}")
    print(f"  Sell signals: {len([s for s in signals if s['action'] == 'sell'])}")

    # =========================================================================
    # 7. EXECUTE TRADES VIA ALPACA (model-driven)
    # =========================================================================
    print("\nExecuting trades via Alpaca...")
    current_prices = {s: p["current_close"] for s, p in predictions.items()}
    trade_actions = []

    broker = _AlpacaBroker(
        os.environ.get("KRONOS_ALPACA_KEY_ID", ""),
        os.environ.get("KRONOS_ALPACA_SECRET_KEY", ""),
        DB_PATH,
    )

    # Preflight: verify Alpaca credentials are valid before placing trades
    try:
        broker.is_market_open()
    except Exception as e:
        raise RuntimeError(f"Alpaca credential validation failed: {e}") from e

    # Fresh start: reset Alpaca + clear local DB on first run or new account
    alpaca_positions = broker.get_positions()
    has_old_trades = False
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        stale_count = conn.execute("SELECT COUNT(*) FROM trades WHERE price = 0 OR price IS NULL").fetchone()[0]
        if stale_count > 0:
            print(f"Cleaning {stale_count} stale trades (price=0 from broken logging period)...")
            conn.execute("DELETE FROM trades WHERE price = 0 OR price IS NULL")
        trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        has_old_trades = trade_count > 0

    if not alpaca_positions and has_old_trades:
        print("Fresh start — Alpaca has no positions but DB has old trades. Resetting...")
        broker.reset_account()
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM portfolio_snapshots")
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM state")
        print("Fresh start — Alpaca reset + local DB cleared")

    try:
        trade_actions = broker.manage_positions(
            predictions, current_prices,
            max_positions=50,
            min_confidence=0.05,
        )
    except Exception as e:
        print(f"ERROR in trading: {type(e).__name__}: {e}")
        print("Continuing to email report with pre-trade state...")

    # =========================================================================
    # 8. TAKE SNAPSHOT & SEND EMAIL
    # =========================================================================
    summary = broker.get_summary(current_prices, num_scanned=len(valid_data))
    summary["model_params"] = f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M"
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
    elif not send_email:
        print("Email sending disabled for this run (pre-market)")
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
# ALPACA BROKER
# =============================================================================
class _AlpacaBroker:
    """Alpaca paper trading broker. Source of truth for positions and cash."""

    def __init__(self, api_key, secret_key, db_path):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        if not api_key or not secret_key:
            raise ValueError(
                f"Alpaca credentials missing — "
                f"KRONOS_ALPACA_KEY_ID={'SET' if api_key else 'MISSING'}, "
                f"KRONOS_ALPACA_SECRET_KEY={'SET' if secret_key else 'MISSING'}. "
                f"Recreate: modal secret create kronos-alpaca KRONOS_ALPACA_KEY_ID=<key> KRONOS_ALPACA_SECRET_KEY=<secret>"
            )
        self.client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
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

    def reset_account(self):
        """Close all positions and cancel all orders. Fresh start."""
        try:
            self.client.close_all_positions(cancel_orders=True)
            print("  Closed all Alpaca positions and cancelled orders")
        except Exception as e:
            print(f"  Reset note: {e}")

    def is_market_open(self):
        """Check if US stock market is currently open."""
        clock = self.client.get_clock()
        return clock.is_open

    def get_account(self):
        """Get Alpaca account info."""
        account = self.client.get_account()
        return {
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_positions(self):
        """Get all open positions from Alpaca."""
        positions = self.client.get_all_positions()
        result = {}
        for p in positions:
            result[p.symbol] = {
                "qty": float(p.qty),
                "side": p.side.value,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "market_value": float(p.market_value),
            }
        return result

    def submit_order(self, symbol, qty, side, reason="model"):
        """Place market order via Alpaca. Returns True if order submitted."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = self.client.submit_order(order_data)
            import time as _time
            _time.sleep(0.3)
            fill_price = 0.0
            fill_qty = float(qty)
            try:
                all_pos = self.client.get_all_positions()
                for p in all_pos:
                    if p.symbol == symbol:
                        fill_price = float(p.avg_entry_price) if side == "buy" else float(p.current_price)
                        fill_qty = float(p.qty)
                        break
            except Exception:
                pass
            if fill_price == 0.0:
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
            if fill_qty == 0.0:
                fill_qty = float(order.filled_qty) if order.filled_qty else float(qty)
            self._log_trade(symbol, side, fill_qty, fill_price, reason)
            print(f"  ORDER {side.upper()} {fill_qty} x {symbol} @ ${fill_price:.2f}  ({reason})")
            return True
        except Exception as e:
            print(f"  ORDER FAILED {symbol} {side} {qty}: {e}")
            return False

    def close_position(self, symbol, reason="model"):
        """Close entire position (long or short) via Alpaca."""
        try:
            all_positions = self.client.get_all_positions()
            pos = None
            for p in all_positions:
                if p.symbol == symbol:
                    pos = p
                    break
            if pos is None:
                print(f"  CLOSE SKIPPED {symbol}: no open position on Alpaca")
                return False
            shares = float(pos.qty)
            price = float(pos.current_price)
            self.client.close_position(symbol)
            import time as _time
            _time.sleep(0.5)
            verify_positions = self.client.get_all_positions()
            still_open = any(p.symbol == symbol for p in verify_positions)
            if still_open:
                print(f"  CLOSE PENDING {symbol}: order submitted but position still open on Alpaca")
                return False
            self._log_trade(symbol, "sell", shares, price, reason)
            print(f"  CLOSE {symbol} {shares} x @ ${price:.2f}  ({reason})")
            return True
        except Exception as e:
            print(f"  CLOSE FAILED {symbol}: {e}")
            return False

    def _log_trade(self, symbol, action, shares, price, reason):
        """Log trade to SQLite for history."""
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute(
                "INSERT INTO trades (symbol, action, shares, price, total, timestamp, reason) VALUES (?,?,?,?,?,?,?)",
                (symbol, action, shares, price, price * shares if shares and price else 0, datetime.now().isoformat(), reason),
            )

    def manage_positions(self, predictions, current_prices, max_positions=50, min_confidence=0.005):
        """Model-driven: sell where model predicts negative, buy top predictions."""
        actions = []

        # Check if market is open — skip order placement if closed
        if not self.is_market_open():
            print("  Market is CLOSED — skipping order placement (will run on next market day)")
            return actions

        positions = self.get_positions()
        account = self.get_account()
        buy_power = account["buying_power"]

        # SELL: model predicts negative return or no prediction available
        for sym, pos in list(positions.items()):
            pred = predictions.get(sym)
            if pred is None:
                if self.close_position(sym, "no_prediction"):
                    actions.append(f"SELL {sym} (no prediction)")
                continue
            ret = pred.get("predicted_return", 0)
            conf = pred.get("confidence", 0)
            if ret < 0 and conf >= min_confidence:
                if self.close_position(sym, f"model_sell ret={ret:+.2%} conf={conf:.3f}"):
                    actions.append(f"SELL {sym} (ret={ret:+.2%})")

        # BUY: rank by return * confidence
        positions = self.get_positions()
        current_count = len(positions)
        slots = max_positions - current_count

        candidates = []
        for sym, pred in predictions.items():
            if sym in positions:
                continue
            ret = pred.get("predicted_return", 0)
            conf = pred.get("confidence", 0)
            pred_std_pct = pred.get("pred_std_pct", 0)
            if ret <= 0 or conf < min_confidence or pred_std_pct > 0.08:
                continue
            candidates.append((sym, pred, ret * conf))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for sym, pred, score in candidates[:slots]:
            if buy_power < 100:
                break
            # Use real-time price from Alpaca instead of stale historical close
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                from alpaca.data.enums import DataFeed
                quote_req = StockLatestQuoteRequest(symbol_or_symbols=sym, feed=DataFeed.IEX)
                quote = self.data_client.get_stock_latest_quote(quote_req)
                price = float(quote[sym].ask_price) if quote and sym in quote else current_prices.get(sym, pred.get("current_close", 0))
            except Exception:
                price = current_prices.get(sym, pred.get("current_close", 0))
            if price <= 1.0:
                continue
            conf = pred.get("confidence", 0)
            ret = pred.get("predicted_return", 0)

            # Model-driven allocation: score determines position size (up to 20% of buying power)
            position_pct = min(score / (score + 1.0), 0.20)
            position_value = buy_power * position_pct
            position_value = min(position_value, buy_power * 0.95)
            shares = int(position_value / price)

            if shares > 0:
                if self.submit_order(sym, shares, "buy", f"model_buy ret={ret:+.2%} conf={conf:.3f}"):
                    actions.append(f"BUY {sym} x{shares} (ret={ret:+.2%}, conf={conf:.3f})")
                    buy_power = self.get_account()["buying_power"]

        return actions

    def get_summary(self, prices, num_scanned=0):
        """Build summary dict from Alpaca account + SQLite trade log."""
        account = self.get_account()
        positions = self.get_positions()
        today = datetime.now().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path, timeout=30) as conn:
            prev = conn.execute("SELECT total_value FROM portfolio_snapshots ORDER BY date DESC LIMIT 1").fetchone()
            prev_total = float(prev[0]) if prev else 100_000.0

            today_trades = conn.execute(
                "SELECT symbol,action,shares,price,total,timestamp,reason FROM trades WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{today}%",)
            ).fetchall()
            today_buys = [t for t in today_trades if t[1] == "buy"]
            today_sells = [t for t in today_trades if t[1] == "sell"]

            total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            recent = conn.execute(
                "SELECT symbol,action,shares,price,total,timestamp,reason FROM trades ORDER BY timestamp DESC LIMIT 10"
            ).fetchall()

            cash = account["cash"]
            portfolio_value = account["portfolio_value"]
            daily_pnl = portfolio_value - prev_total

            first_snap = conn.execute("SELECT total_value FROM portfolio_snapshots ORDER BY date ASC LIMIT 1").fetchone()
            initial_cash = float(first_snap[0]) if first_snap else 100_000.0
            cum_pnl = portfolio_value - initial_cash

            conn.execute(
                "INSERT OR REPLACE INTO portfolio_snapshots (date,cash,equity,total_value,positions_json,daily_pnl,cumulative_pnl) VALUES (?,?,?,?,?,?,?)",
                (today, cash, portfolio_value, portfolio_value, json.dumps(positions), daily_pnl, cum_pnl),
            )

            # Realized P&L from trade log — match buy/sell pairs per symbol
            all_trades = conn.execute(
                "SELECT symbol,action,shares,price,timestamp FROM trades ORDER BY timestamp"
            ).fetchall()
            buy_costs = {}
            total_realized = 0
            realized_today = 0
            for t in all_trades:
                sym, action, shares, price, ts = t
                if action == "buy":
                    prev_bc = buy_costs.get(sym, {"shares": 0, "total_cost": 0})
                    new_shares = prev_bc["shares"] + shares
                    new_cost = prev_bc["total_cost"] + (price * shares if shares and price else 0)
                    buy_costs[sym] = {"shares": new_shares, "total_cost": new_cost}
                elif action == "sell" and sym in buy_costs:
                    bc = buy_costs[sym]
                    if bc["shares"] > 0 and shares > 0 and price > 0:
                        avg_cost = bc["total_cost"] / bc["shares"]
                        pnl = (price - avg_cost) * shares
                        total_realized += pnl
                        if ts.startswith(today):
                            realized_today += pnl
                        bc["shares"] -= shares
                        bc["total_cost"] = avg_cost * max(0, bc["shares"])

            first_snap = conn.execute("SELECT total_value FROM portfolio_snapshots ORDER BY date ASC LIMIT 1").fetchone()
            initial_cash = float(first_snap[0]) if first_snap else 100_000.0

        # Position details from Alpaca
        pos_details = []
        for sym, pos in positions.items():
            side = pos.get("side", "long")
            qty = abs(pos.get("qty", 0))
            avg_cost = pos.get("avg_entry_price", 0)
            current_price = pos.get("current_price", avg_cost)
            pnl = pos.get("unrealized_pl", 0)
            pnl_pct = pos.get("unrealized_plpc", 0) * 100
            pos_details.append({
                "symbol": sym, "shares": qty, "side": side,
                "avg_cost": avg_cost, "entry_date": today,
                "current_price": current_price, "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
        pos_details.sort(key=lambda x: x["pnl"], reverse=True)

        return {
            "date": today, "cash": cash, "equity": portfolio_value, "total_value": portfolio_value,
            "buying_power": account["buying_power"],
            "long_market_value": account["long_market_value"],
            "short_market_value": account["short_market_value"],
            "daily_pnl": daily_pnl, "cumulative_pnl": cum_pnl,
            "num_positions": len(positions),
            "total_trades": total_trades, "recent_trades": recent,
            "num_scanned": num_scanned,
            "today_buys": today_buys, "today_sells": today_sells,
            "realized_pnl_today": realized_today,
            "total_realized": total_realized,
            "pos_details": pos_details,
            "initial_cash": initial_cash,
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
    return symbols[:1000]


def _fetch_ohlcv_batch(api_keys, symbols, requests_mod, pd_mod, outputsize=600):
    """Fetch OHLCV data with key rotation."""
    import threading
    results = {}
    key_idx = [0]
    call_times = {k: [] for k in api_keys}
    lock = threading.Lock()

    def get_key():
        while True:
            with lock:
                key = api_keys[key_idx[0] % len(api_keys)]
                key_idx[0] += 1
                now = time.time()
                call_times[key] = [t for t in call_times[key] if now - t < 60]
                if len(call_times[key]) < 8:
                    call_times[key].append(time.time())
                    return key
                sleep_time = 60 - (now - call_times[key][0]) + 0.1
            time.sleep(max(sleep_time, 0.5))

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
        current_close = pred.get("current_close", 0)

        # Skip penny stocks and high-noise predictions
        if current_close < 1.0 or confidence < 0.005 or pred_std_pct > 0.08:
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
    lines = []

    # Header
    lines.append(f"Date: {summary['date']}")
    lines.append("Broker: Alpaca Paper Trading (2x leverage)")
    lines.append("")

    # AI Runtime
    lines.append("AI RUNTIME")
    lines.append("-" * 60)
    lines.append("Backend Used: modal (T4 GPU)")
    lines.append(f"Model Used: Kronos-base ({summary.get('model_params', '~124')} params)")
    lines.append("Status: OK")
    lines.append("")

    # Portfolio Summary
    lines.append("PORTFOLIO SUMMARY")
    lines.append("-" * 60)
    lines.append(f"Stocks Scanned Today: {summary.get('num_scanned', 0)}")
    lines.append(f"Open Positions: {summary['num_positions']}")
    lines.append(f"Positions Closed Today: {len(summary.get('today_sells', []))}")
    lines.append(f"New Positions Opened Today: {len(summary.get('today_buys', []))}")
    lines.append(f"Portfolio Value: ${summary['equity']:,.2f}")
    lines.append(f"Buying Power (2x): ${summary['buying_power']:,.2f}")
    lines.append(f"Long Exposure: ${summary['long_market_value']:,.2f}")
    lines.append(f"Short Exposure: ${summary['short_market_value']:,.2f}")
    lines.append(f"Cash: ${summary['cash']:,.2f}")
    lines.append("")

    # Daily Performance
    lines.append("DAILY PERFORMANCE")
    lines.append("-" * 60)
    initial = summary['initial_cash']
    daily_pct = (summary['daily_pnl'] / initial) * 100
    lines.append(f"Realized P&L (Today): {daily_pct:+.2f}% (${summary['daily_pnl']:,.2f})")
    lines.append("")

    # Account Totals
    lines.append("ACCOUNT TOTALS")
    lines.append("-" * 60)
    total_realized_pct = (summary.get('total_realized', 0) / initial) * 100
    lines.append(f"Total Realized P&L (Lifetime): {total_realized_pct:+.2f}% (${summary.get('total_realized', 0):,.2f})")
    unrealized = sum(p["pnl"] for p in summary.get('pos_details', []))
    unrealized_pct = (unrealized / initial) * 100
    lines.append(f"Unrealized P&L (Open Positions): {unrealized_pct:+.2f}% (${unrealized:,.2f})")
    lines.append("-" * 60)
    total_return = (summary.get('total_realized', 0) + unrealized) / initial * 100
    lines.append(f"TOTAL ACCOUNT RETURN: {total_return:+.2f}% (Lifetime Realized + Unrealized)")
    lines.append("")

    # Positions Entered Today
    today_buys = summary.get('today_buys', [])
    if today_buys:
        lines.append("POSITIONS ENTERED TODAY")
        lines.append("-" * 60)
        lines.append(f"{'Symbol':<8} | {'Side':<6} | {'Entry':>8} | {'Alloc %':>8} | {'Alloc $':>12} | Reason")
        lines.append("-" * 60)
        for t in today_buys:
            sym, action, shares, price, total, ts, reason = t
            alloc_pct = (total / initial) * 100 if initial else 0
            lines.append(f"{sym:<8} | {'LONG':<6} | ${price:>7.2f} | {alloc_pct:>7.1f}% | ${total:>11,.2f} | {reason}")
        lines.append("")

    # Positions Closed Today
    today_sells = summary.get('today_sells', [])
    if today_sells:
        lines.append("POSITIONS CLOSED TODAY")
        lines.append("-" * 60)
        lines.append(f"{'Symbol':<8} | {'Side':<6} | {'Exit':>8} | Reason")
        lines.append("-" * 60)
        for t in today_sells:
            sym, action, shares, price, total, ts, reason = t
            lines.append(f"{sym:<8} | {'LONG':<6} | ${price:>7.2f} | {reason}")
        lines.append("")

    # Open Positions (Unrealized)
    pos_details = summary.get('pos_details', [])
    if pos_details:
        lines.append("OPEN POSITIONS (Unrealized)")
        lines.append("-" * 60)
        lines.append(f"{'Symbol':<8} | {'Side':<6} | {'Entry':>8} | {'Current':>8} | {'P&L %':>8} | {'P&L $':>10}")
        lines.append("-" * 60)
        for p in pos_details:
            side_label = "SHORT" if p.get("side") == "short" else "LONG"
            lines.append(
                f"{p['symbol']:<8} | {side_label:<6} | ${p['avg_cost']:>7.2f} | ${p['current_price']:>7.2f} | {p['pnl_pct']:>+7.2f}% | ${p['pnl']:>+9.2f}"
            )
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def _send_email(subject, body, sender_email, sender_password, recipient_email):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
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
    try:
        with open(ctx_path, "w") as f:
            json.dump(context, f, indent=2)
        print(f"Context written for next run")
    except Exception as e:
        print(f"WARNING: Failed to write context file: {e}")


# =============================================================================
# CRON SCHEDULES
# =============================================================================
@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
        modal.Secret.from_name("kronos-email"),
        modal.Secret.from_name("kronos-alpaca"),
    ],
    schedule=modal.Cron("30 13 * * 1-5"),
    timeout=3600,
    gpu="T4",
)
def pre_market_run():
    """Pre-market: predict + trade, 9:30 AM ET weekdays."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    if now_et.hour != 9 or now_et.minute != 30:
        print(f"Skipping pre_market_run: ET time is {now_et.strftime('%H:%M')}, expected 09:30")
        return
    try:
        run_trading_cycle.local()
    except Exception as e:
        print(f"FATAL ERROR in pre_market_run: {type(e).__name__}: {e}")
        try:
            sender = os.environ.get("KRONOS_EMAIL", "")
            password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
            recipient = os.environ.get("KRONOS_RECIPIENT", sender)
            if sender and password and recipient:
                _send_email(
                    subject=f"Kronos ERROR — pre_market_run — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    body=f"FATAL ERROR in pre-market run:\n\n{type(e).__name__}: {e}\n\nCheck Modal logs for details.",
                    sender_email=sender,
                    sender_password=password,
                    recipient_email=recipient,
                )
        except Exception:
            pass
        raise


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
        modal.Secret.from_name("kronos-email"),
        modal.Secret.from_name("kronos-alpaca"),
    ],
    schedule=modal.Cron("45 19 * * 1-5"),
    timeout=3600,
    gpu="T4",
)
def post_market_run():
    """Post-market: predict + rebalance + email report, 3:45 PM ET weekdays."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    if now_et.hour != 15 or now_et.minute != 45:
        print(f"Skipping post_market_run: ET time is {now_et.strftime('%H:%M')}, expected 15:45")
        return
    run_trading_cycle.local(send_email=True)


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
    ],
    schedule=modal.Cron("0 12 * * 1-5"),
    timeout=3600,
)
def update_data_morning():
    """Update latest bars for all cached stocks. Runs daily 8:00 AM ET before pre-market."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    if now_et.hour != 8:
        print(f"Skipping update_data_morning: ET time is {now_et.strftime('%H:%M')}, expected 08:00")
        return
    try:
        _run_data_update()
    except Exception as e:
        print(f"FATAL ERROR in update_data_morning: {type(e).__name__}: {e}")
        try:
            sender = os.environ.get("KRONOS_EMAIL", "")
            password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
            recipient = os.environ.get("KRONOS_RECIPIENT", sender)
            if sender and password and recipient:
                _send_email(
                    subject=f"Kronos ERROR — data update failed — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    body=f"FATAL ERROR in morning data update:\n\n{type(e).__name__}: {e}\n\nCheck Modal logs for details.",
                    sender_email=sender,
                    sender_password=password,
                    recipient_email=recipient,
                )
        except Exception:
            pass
        raise


@app.function(
    image=image,
    volumes={VOLUME_PATH: vol},
    secrets=[
        modal.Secret.from_name("kronos-twelve-data"),
    ],
    schedule=modal.Cron("0 19 * * 1-5"),
    timeout=3600,
)
def update_data_afternoon():
    """Update latest bars for all cached stocks. Runs daily 3:00 PM ET before post-market."""
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    if now_et.hour != 15:
        print(f"Skipping update_data_afternoon: ET time is {now_et.strftime('%H:%M')}, expected 15:00")
        return
    try:
        _run_data_update()
    except Exception as e:
        print(f"FATAL ERROR in update_data_afternoon: {type(e).__name__}: {e}")
        try:
            sender = os.environ.get("KRONOS_EMAIL", "")
            password = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
            recipient = os.environ.get("KRONOS_RECIPIENT", sender)
            if sender and password and recipient:
                _send_email(
                    subject=f"Kronos ERROR — data update failed — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    body=f"FATAL ERROR in afternoon data update:\n\n{type(e).__name__}: {e}\n\nCheck Modal logs for details.",
                    sender_email=sender,
                    sender_password=password,
                    recipient_email=recipient,
                )
        except Exception:
            pass
        raise


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
