# src/maktsprak_pipeline/logger.py
from loguru import logger
from .config import LOG_LEVEL
from datetime import datetime, timezone
from pathlib import Path
import sys

# -----------------------------
# Setup log directory
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Configure logger
# -----------------------------
logger.remove()
log_filename = LOG_DIR / f"etl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_filename, level=LOG_LEVEL, rotation="1 MB", encoding="utf-8")
logger.add(sys.stdout, level=LOG_LEVEL)  # fortfarande till terminal

def get_logger():
    return logger
