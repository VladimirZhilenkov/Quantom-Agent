"""Execute ORCA via subprocess and return QChemResult."""

from schemas import QChemJob, QChemResult


def run_orca(job: QChemJob) -> QChemResult:
    """Run a quantum chemistry calculation using ORCA."""
    raise NotImplementedError
