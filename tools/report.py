"""Jinja2 report generator → markdown."""

from schemas import QChemResult


def generate_report(results: list[QChemResult], title: str) -> str:
    """Generate a human-readable markdown report from calculation results."""
    raise NotImplementedError
