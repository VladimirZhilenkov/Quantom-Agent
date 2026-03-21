"""cclib-based output parser → QChemResult."""

from schemas import QChemResult


def parse_output(filepath: str) -> QChemResult:
    """Parse QM engine output file using cclib."""
    raise NotImplementedError
