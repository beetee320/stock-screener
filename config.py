from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

CACHE_DATA = True
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

STATIC_DATA_FILE = Path("static_data.csv")

BATCH_SIZE = 50
MAX_RETRIES = 3

# Market Cap Classification Thresholds
LARGE_CAP_THRESHOLD = 20000
MID_CAP_THRESHOLD = 5000

# Score Configuration
SCORE_CONFIG = {
    'short_term': {
        'early_detection_weight': 0.5,
        'momentum_weight': 0.3,
        'trend_weight': 0.2
    },
    'balanced': {
        'early_detection_weight': 0.3,
        'momentum_weight': 0.35,
        'trend_weight': 0.35
    },
    'long_term': {
        'early_detection_weight': 0.2,
        'momentum_weight': 0.3,
        'trend_weight': 0.5
    }
}
