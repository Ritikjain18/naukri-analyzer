import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "naukri.db"
MODEL_FAST = "openai/gpt-oss-20b"
MODEL_SMART = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"  # gpt-oss are reasoning models; hidden reasoning tokens count against TPM
ROW_CAP = 200
HISTORY_TURNS = 6
DESCRIBE_COLUMNS = 20
USAGE_DB_PATH = ROOT / "data" / "usage.db"
CONTEXT_WINDOW = 131072
OUTPUT_RESERVE = 1500  # reasoning tokens count as output
SAFETY_MARGIN = 0.10
RATE_HEADROOM = 0.9  # fraction of a model's TPM/TPD a single call may use
SAMPLE_TOKENS = 1500
# Measured from Groq response headers for gpt-oss models on this account: 8,000 TPM, 1,000 requests/day;
# tpd is an assumption. The two models may share one rate-limit bucket; that is not modelled here.
MODEL_LIMITS = {
    MODEL_SMART: {"tpm": 8000, "tpd": 200000},
    MODEL_FAST: {"tpm": 8000, "tpd": 200000},
}
JUDGE_ENABLED = True
JUDGE_RETRIEVAL = True
JUDGE_MIN_SCORE = 3
MAX_REJECTIONS = 2
MAX_GUARD_FAILURES = 3
MEMORY_SLICE_ROWS = 50
APP_DB_PATH = ROOT / "data" / "app.db"
MEMORY_DIR = ROOT / "memory"
LOCKOUT_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
MIN_PASSWORD_LENGTH = 8
PRIOR_SESSIONS = 3
PRIOR_CONTEXT_TOKENS = 600
ABANDONED_SESSION_MINUTES = 30
MAX_SUMMARIES_PER_LOGIN = 2


class MissingKeyError(RuntimeError):
    pass


def bootstrap_code() -> str:
    """Optional setup code required to create the first admin (set BOOTSTRAP_CODE on hosted deployments)."""
    return os.environ.get("BOOTSTRAP_CODE", "").strip()


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise MissingKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
