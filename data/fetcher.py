import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class TwelveDataFetcher:
    """Fetches OHLCV data from Twelve Data with 8-key round-robin rotation."""

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_keys: list[str]):
        self.api_keys = api_keys
        self.key_index = 0
        self.key_call_times = {k: [] for k in api_keys}
        self.min_interval = 7.5  # 8 credits/min = 1 call per 7.5s per key

    def _get_next_key(self) -> str:
        """Round-robin with rate limiting per key."""
        key = self.api_keys[self.key_index % len(self.api_keys)]
        self.key_index += 1

        now = time.time()
        self.key_call_times[key] = [
            t for t in self.key_call_times[key] if now - t < 60
        ]
        if len(self.key_call_times[key]) >= 8:
            sleep_time = 60 - (now - self.key_call_times[key][0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.key_call_times[key].append(time.time())
        return key

    def fetch_stock_list(self, exchange: str = "NASDAQ,AMEX,NYSE") -> list[str]:
        """Fetch top US stocks by market cap from Twelve Data."""
        key = self._get_next_key()
        try:
            resp = requests.get(
                f"{self.BASE_URL}/stocks",
                params={"exchange": exchange, "apikey": key},
                timeout=30,
            )
            data = resp.json().get("data", [])
            symbols = [s["symbol"] for s in data if s.get("symbol")]
            return symbols[:1200]  # Grab extra, we'll trim to 1000
        except Exception as e:
            print(f"Error fetching stock list: {e}")
            return []

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 600,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for a single stock."""
        key = self._get_next_key()
        try:
            resp = requests.get(
                f"{self.BASE_URL}/time_series",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "outputsize": outputsize,
                    "apikey": key,
                },
                timeout=30,
            )
            data = resp.json()
            if "values" not in data:
                return None

            df = pd.DataFrame(data["values"])
            df = df.rename(columns={
                "datetime": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def fetch_batch(
        self,
        symbols: list[str],
        interval: str = "1day",
        outputsize: int = 600,
        max_workers: int = 16,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple stocks with parallel requests."""
        results = {}
        total = len(symbols)

        def _fetch(sym):
            return sym, self.fetch_ohlcv(sym, interval, outputsize)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch, s): s for s in symbols}
            done = 0
            for future in as_completed(futures):
                done += 1
                sym, df = future.result()
                if df is not None and len(df) >= 50:
                    results[sym] = df
                if done % 100 == 0:
                    print(f"  Fetched {done}/{total} stocks...")

        print(f"  Successfully fetched {len(results)}/{total} stocks")
        return results

    def fetch_latest_bars(
        self,
        symbols: list[str],
        existing_data: dict[str, pd.DataFrame],
        interval: str = "1day",
        num_bars: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """Fetch only latest bars and append to existing data."""
        updated = {}
        total = len(symbols)

        for i, sym in enumerate(symbols):
            if sym not in existing_data:
                continue

            key = self._get_next_key()
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/time_series",
                    params={
                        "symbol": sym,
                        "interval": interval,
                        "outputsize": num_bars,
                        "apikey": key,
                    },
                    timeout=30,
                )
                data = resp.json()
                if "values" not in data:
                    updated[sym] = existing_data[sym]
                    continue

                new_df = pd.DataFrame(data["values"])
                new_df = new_df.rename(columns={"datetime": "timestamp"})
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
                for col in ["open", "high", "low", "close", "volume"]:
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

                existing = existing_data[sym]
                combined = pd.concat([existing, new_df]).drop_duplicates(
                    subset=["timestamp"], keep="last"
                )
                combined = combined.sort_values("timestamp").reset_index(drop=True)
                combined = combined.tail(600)  # Keep max 600 bars
                updated[sym] = combined

            except Exception:
                updated[sym] = existing_data[sym]

            if (i + 1) % 100 == 0:
                print(f"  Updated {i+1}/{total} stocks...")

        return updated

    def load_cached_data(self, cache_dir: str) -> dict[str, pd.DataFrame]:
        """Load cached OHLCV data from disk."""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return {}

        data = {}
        for f in cache_path.glob("*.parquet"):
            sym = f.stem
            try:
                data[sym] = pd.read_parquet(f)
            except Exception:
                pass
        return data

    def save_cached_data(self, data: dict[str, pd.DataFrame], cache_dir: str):
        """Save OHLCV data to disk cache."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        for sym, df in data.items():
            try:
                df.to_parquet(cache_path / f"{sym}.parquet", index=False)
            except Exception:
                pass
