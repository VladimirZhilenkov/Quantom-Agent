"""Execute Psi4 via Python API and return QChemResult."""

from schemas import QChemJob, QChemResult


def run_psi4(job: QChemJob) -> QChemResult:
    """Run a quantum chemistry calculation using Psi4."""
    raise NotImplementedError
