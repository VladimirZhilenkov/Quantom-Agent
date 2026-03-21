"""Failure classification + retry heuristics."""


def classify_error(error: Exception) -> str:
    """Classify a calculation error for retry strategy selection."""
    raise NotImplementedError


def suggest_fix(error_class: str, job_params: dict) -> dict:
    """Suggest modified parameters to recover from a failed calculation."""
    raise NotImplementedError
