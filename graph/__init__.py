import os

# Disable Guardrails/OpenTelemetry network export before anything can import guardrails.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
