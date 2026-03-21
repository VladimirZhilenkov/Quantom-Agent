"""
schemas.py
----------
Shared Pydantic models — the central data contract for all modules.
All tools, parsers, and DB layers use these models.
"""

from pydantic import BaseModel
from typing import Literal


class QChemJob(BaseModel):
    """Input specification for a quantum chemistry calculation."""

    id: str
    method: str  # e.g. "b3lyp", "ccsd(t)"
    basis: str  # e.g. "def2-tzvp"
    charge: int
    multiplicity: int
    atoms: list[tuple[str, float, float, float]]  # (symbol, x, y, z) in Å
    job_type: Literal["energy", "opt", "freq"]
    engine: Literal["pyscf", "orca", "psi4"] = "pyscf"
    solvent: str | None = None
    nprocs: int = 1
    memory_mb: int = 4000


class QChemResult(BaseModel):
    """Output from a quantum chemistry calculation."""

    job_id: str
    energy: float | None  # total energy in Hartree
    converged: bool
    geometry: list[tuple[str, float, float, float]] | None
    frequencies: list[float] | None  # in cm⁻¹
    dipole: tuple[float, float, float] | None  # Debye
    wall_time: float  # seconds
    engine: str
    method: str
    basis: str
    raw_output: str
    output_path: str | None = None
