"""Fetch OHLCV data for fine-tuning. Run before fine_tune.py."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import _fetch_ohlcv_batch, _fetch_stock_universe
import requests
import pandas as pd

DATA_DIR = Path("data")
OHLCV_DIR = DATA_DIR / "ohlcv"
STOCKS_FILE = DATA_DIR / "stocks.json"


def main():
    keys = json.loads(os.environ["TWELVE_DATA_KEYS"])
    print(f"Loaded {len(keys)} API keys")

    if STOCKS_FILE.exists():
        with open(STOCKS_FILE) as f:
            symbols = json.load(f)
        print(f"Loaded {len(symbols)} symbols from stocks.json")
    else:
        print("Fetching stock universe from Twelve Data...")
        symbols = _fetch_stock_universe(keys, requests)
        STOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STOCKS_FILE, "w") as f:
            json.dump(symbols, f)
        print(f"Cached {len(symbols)} symbols")

    OHLCV_DIR.mkdir(parents=True, exist_ok=True)

    to_fetch = [s for s in symbols if not (OHLCV_DIR / f"{s}.parquet").exists()]
    print(f"Need to fetch {len(to_fetch)} stocks (have {len(symbols) - len(to_fetch)} cached)")

    fetched = 0
    for batch_start in range(0, len(to_fetch), 200):
        batch = to_fetch[batch_start : batch_start + 200]
        print(f"  Fetching batch {batch_start // 200 + 1}: {len(batch)} stocks...")
        data = _fetch_ohlcv_batch(keys, batch, requests, pd, outputsize=900)
        for sym, df in data.items():
            df.to_parquet(OHLCV_DIR / f"{sym}.parquet", index=False)
        fetched += len(data)
        print(f"  Got {len(data)} stocks")

    total = len(list(OHLCV_DIR.glob("*.parquet")))
    print(f"Total: {fetched} new + {total} existing parquet files")


if __name__ == "__main__":
    main()
