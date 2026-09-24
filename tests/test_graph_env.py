import importlib
import os

import graph


def test_otel_disabled_on_import_graph(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    importlib.reload(graph)
    assert os.environ["OTEL_SDK_DISABLED"] == "true"
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
