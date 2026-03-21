"""Execute PySCF scripts and return QChemResult."""

from schemas import QChemJob, QChemResult


def run_pyscf(job: QChemJob) -> QChemResult:
    """Run a quantum chemistry calculation using PySCF."""
    raise NotImplementedError
