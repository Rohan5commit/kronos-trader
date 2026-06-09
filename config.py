import os
import json

# =============================================================================
# Twelve Data API Keys (8 free accounts)
# Set as Modal secret: modal secret create kronos-secrets TWELVE_DATA_KEYS='["key1","key2",...]'
# =============================================================================
TWELVE_DATA_KEYS = json.loads(os.environ.get("TWELVE_DATA_KEYS", "[]"))

# =============================================================================
# Modal Config
# =============================================================================
VOLUME_PATH = "/kronos-data"
MODEL_CACHE_PATH = f"{VOLUME_PATH}/model"
DATA_CACHE_PATH = f"{VOLUME_PATH}/ohlcv"
TRADES_DB_PATH = f"{VOLUME_PATH}/trades.db"
STOCK_LIST_PATH = f"{VOLUME_PATH}/stocks.json"

# =============================================================================
# Kronos Model
# =============================================================================
KRONOS_MODEL_REPO = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
CONTEXT_LENGTH = 512
PREDICT_WINDOW = 10

# =============================================================================
# Data Fetching
# =============================================================================
BARS_TO_FETCH = 600  # Extra buffer beyond context length
BATCH_SIZE_PER_KEY = 8  # Credits per minute per key
NUM_KEYS = 8
FETCH_BATCH_SIZE = 64  # Stocks per API batch (8 keys * 8 credits/min)

# =============================================================================
# Trading Strategy — Autonomous Hedge Fund Manager
# =============================================================================
INITIAL_CASH = 100_000.0
MAX_POSITIONS = 50          # Max stocks held simultaneously
POSITION_SIZE_PCT = 0.02    # 2% of portfolio per position
STOP_LOSS_PCT = 0.05        # 5% stop loss
TAKE_PROFIT_PCT = 0.15      # 15% take profit
REBALANCE_THRESHOLD = 0.02  # Rebalance if drift > 2%
MIN_CONFIDENCE = 0.005      # Min predicted return to enter (0.5%)

# =============================================================================
# Risk Management
# =============================================================================
MAX_DRAWDOWN_PCT = 0.10     # Pause trading if drawdown > 10%
MAX_SECTOR_WEIGHT = 0.20    # Max 20% in any sector
DAILY_LOSS_LIMIT = 0.03     # Stop trading if daily loss > 3%

# =============================================================================
# Email (Gmail SMTP — set later)
# =============================================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.environ.get("KRONOS_EMAIL", "")
EMAIL_PASSWORD = os.environ.get("KRONOS_EMAIL_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("KRONOS_RECIPIENT", "")

# =============================================================================
# Schedule (Eastern Time)
# =============================================================================
PRE_MARKET_RUN = "30 9 * * 1-5"   # 9:30 AM ET, weekdays
POST_MARKET_RUN = "45 15 * * 1-5" # 3:45 PM ET, weekdays
