import math

from config import CONTEXT_WINDOW, MODEL_LIMITS, OUTPUT_RESERVE, SAFETY_MARGIN

_encode_len = None  # callable(text) -> int; None = lazy tiktoken
_enc = None
_enc_failed = False


def use_estimator(fn) -> None:
    """Tests inject a local estimator so tiktoken's first-use download never runs."""
    global _encode_len
    _encode_len = fn


def _tiktoken_len(text: str) -> int:
    global _enc, _enc_failed
    if _enc is None and not _enc_failed:
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc_failed = True
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    return math.ceil(len(text) / 3)


def count_tokens(text: str) -> int:
    return math.ceil((_encode_len or _tiktoken_len)(text) * (1 + SAFETY_MARGIN))


def effective_limit(model: str) -> int:
    return min(CONTEXT_WINDOW, MODEL_LIMITS.get(model, {}).get("tpm", CONTEXT_WINDOW))


def available_tokens(model: str, system: str, history: str, question: str) -> int:
    used = count_tokens(system) + count_tokens(history) + count_tokens(question)
    return max(0, effective_limit(model) - OUTPUT_RESERVE - used)


def clip_to_tokens(text: str, budget: int) -> str:
    if count_tokens(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid] + "...") <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "..."


def fit_rows(df, budget: int, min_rows: int = 5) -> str:
    total = len(df)
    text = df.to_csv(index=False)
    if count_tokens(text) <= budget:
        return text
    note_cost = count_tokens(f"# truncated: showing {total} of {total} rows\n")
    lo, hi = min(min_rows, total), total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(df.head(mid).to_csv(index=False)) + note_cost <= budget:
            lo = mid
        else:
            hi = mid - 1
    return df.head(lo).to_csv(index=False) + f"# truncated: showing {lo} of {total} rows\n"
