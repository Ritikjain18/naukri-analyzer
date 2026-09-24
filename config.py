import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "naukri.db"
MODEL_FAST = "llama-3.1-8b-instant"
MODEL_SMART = "llama-3.3-70b-versatile"
ROW_CAP = 200
HISTORY_TURNS = 6
DESCRIBE_COLUMNS = 20
USAGE_DB_PATH = ROOT / "data" / "usage.db"
CONTEXT_WINDOW = 131072
OUTPUT_RESERVE = 1000
SAFETY_MARGIN = 0.10
SAMPLE_TOKENS = 1500
# Groq free-tier assumptions; verify in the Groq console and edit here if they differ.
MODEL_LIMITS = {
    MODEL_SMART: {"tpm": 12000, "tpd": 100000},
    MODEL_FAST: {"tpm": 6000, "tpd": 500000},
}
JUDGE_ENABLED = True
JUDGE_RETRIEVAL = True
JUDGE_MIN_SCORE = 3
MAX_REJECTIONS = 2
MAX_GUARD_FAILURES = 3
MEMORY_SLICE_ROWS = 50


class MissingKeyError(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise MissingKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
