import io
import json
from pathlib import Path

import pandas as pd


def parse_upload(name: str, content: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(content))
    if suffix == ".json":
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if len(lists) == 1 else [data]
        return pd.json_normalize(data)
    raise ValueError(f"Unsupported file type: {suffix or name}")
