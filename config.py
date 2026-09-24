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


class MissingKeyError(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise MissingKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
