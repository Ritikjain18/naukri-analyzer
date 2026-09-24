"""Optional Guardrails AI harness around the pure checks in graph.guards.

The pure ``check_*`` functions are the source of truth. When ``guardrails-ai`` is importable
they are executed as custom Guardrails validators; the returned GuardResult is always the one
the pure check produced. Guardrails telemetry (OTLP export to a remote endpoint) is disabled
before the library is imported so no network calls are made.
"""
import logging
import os
from typing import Callable

from graph.guards import GuardResult, check_input, check_insight

os.environ.setdefault("OTEL_SDK_DISABLED", "true")  # must precede the guardrails/opentelemetry import

try:  # guardrails-ai is optional; the pure checks are the source of truth
    import guardrails  # noqa: F401
    from guardrails import Guard, Validator, register_validator
    from guardrails.settings import settings as _gr_settings
    from guardrails.validators import FailResult, PassResult

    _gr_settings.rc.enable_metrics = False
    GUARDRAILS_AVAILABLE = True
except Exception:
    GUARDRAILS_AVAILABLE = False

logger = logging.getLogger(__name__)
HARNESS_RUNS = 0  # number of times the Guardrails validator actually executed
_VALIDATORS: dict[str, type] = {}


def _validator_class(name: str):
    if name not in _VALIDATORS:

        @register_validator(name=name, data_type="string")
        class _CheckValidator(Validator):
            def _validate(self, value, metadata):
                result: GuardResult = metadata["check"](*metadata["args"], **metadata["kwargs"])
                metadata["result"] = result
                global HARNESS_RUNS
                HARNESS_RUNS += 1
                if result.ok:
                    return PassResult()
                return FailResult(error_message="; ".join(result.reasons))

        _VALIDATORS[name] = _CheckValidator
    return _VALIDATORS[name]


def _run_with_guardrails(name: str, check: Callable[..., GuardResult], subject, **kwargs) -> GuardResult:
    pure = check(subject, **kwargs)
    try:
        guard = Guard().use(_validator_class(name)(on_fail="noop"))
        guard.configure(allow_metrics_collection=False)
        meta = {"check": check, "args": (subject,), "kwargs": kwargs}
        guard.validate(str(subject), metadata=meta)
    except Exception:  # harness failure: the pure result stands
        logger.debug("Guardrails harness failed for %s; using pure result", name, exc_info=True)
    return pure


def run_input_guard(question: str, schema_text: str = "", revision: bool = False) -> GuardResult:
    if not GUARDRAILS_AVAILABLE:
        return check_input(question, schema_text, revision)
    return _run_with_guardrails("naukri/hr_input", check_input, question, schema_text=schema_text, revision=revision)


def run_output_guard(insight, data_slice, question: str = "") -> GuardResult:
    if not GUARDRAILS_AVAILABLE:
        return check_insight(insight, data_slice, question)
    return _run_with_guardrails("naukri/insight_output", check_insight, insight, data_slice=data_slice, question=question)
