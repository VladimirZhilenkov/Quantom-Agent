"""Execute a PySCF script in a subprocess and return a structured QChemResult."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from schemas import QChemResult


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------

class SCFConvergenceError(Exception):
    """Raised when the SCF procedure fails to converge."""

    def __init__(self, message: str, partial_result: QChemResult | None = None):
        super().__init__(message)
        self.partial_result = partial_result


class PySCFTimeoutError(Exception):
    """Raised when PySCF execution exceeds the time limit."""


class PySCFOOMError(Exception):
    """Raised when the process is killed due to memory exhaustion."""


# ---------------------------------------------------------------------------
# Results collector (appended to every script before execution)
# ---------------------------------------------------------------------------

_MARKER_START = "<<<PYSCF_RESULTS>>>"
_MARKER_END = "<<<END_PYSCF_RESULTS>>>"

_COLLECTOR = """
# --- pyscf wrapper: results collector ---
import json as _json

_g = globals()
_res = {"converged": True}

# Energy and convergence from the main computational object
for _v in ("mf", "mf_sol", "mc", "mycc", "mymp", "mp2", "pt"):
    _o = _g.get(_v)
    if _o is not None and hasattr(_o, "e_tot"):
        _res["energy"] = float(_o.e_tot)
        if hasattr(_o, "converged"):
            _res["converged"] = bool(_o.converged)
        break

# Molecule info and geometry (prefer optimised geometry mol_eq)
_mol = _g.get("mol_eq") or _g.get("mol")
if _mol is not None:
    try:
        _b = _mol.basis
        _res["basis"] = _b if isinstance(_b, str) else ""
    except Exception:
        pass
    try:
        _c = _mol.atom_coords(unit="ANG")
        _res["geometry"] = [
            [_mol.atom_symbol(i), float(_c[i][0]), float(_c[i][1]), float(_c[i][2])]
            for i in range(_mol.natm)
        ]
    except Exception:
        pass

# Frequencies
for _fv in ("freq", "freqs", "frequencies"):
    _fd = _g.get(_fv)
    if _fd is not None:
        try:
            _res["frequencies"] = [
                float(f)
                for f in (_fd.tolist() if hasattr(_fd, "tolist") else list(_fd))
            ]
        except Exception:
            pass
        break

# Dipole moment
for _dv in ("dip", "dipole"):
    _dd = _g.get(_dv)
    if _dd is not None:
        try:
            _d = _dd.tolist() if hasattr(_dd, "tolist") else list(_dd)
            if len(_d) == 3:
                _res["dipole"] = [float(_d[0]), float(_d[1]), float(_d[2])]
        except Exception:
            pass
        break

print("<<<PYSCF_RESULTS>>>")
print(_json.dumps(_res))
print("<<<END_PYSCF_RESULTS>>>")
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pyscf(script: str, timeout: int = 3600) -> QChemResult:
    """Execute a PySCF Python script and return a structured result.

    Parameters
    ----------
    script : str
        Complete PySCF Python script to execute.
    timeout : int
        Maximum wall-time in seconds (default 3600).

    Returns
    -------
    QChemResult

    Raises
    ------
    SCFConvergenceError
        If SCF fails to converge (carries a *partial_result*).
    PySCFTimeoutError
        If execution exceeds *timeout* seconds.
    PySCFOOMError
        If the process is killed due to memory exhaustion.
    RuntimeError
        If the process exits with a non-zero code for other reasons.
    """
    job_id = uuid.uuid4().hex[:12]
    full_script = script + "\n" + _COLLECTOR

    with tempfile.TemporaryDirectory(prefix="pyscf_") as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text(full_script, encoding="utf-8")

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            raise PySCFTimeoutError(
                f"PySCF execution timed out after {timeout}s"
            ) from exc

        wall_time = time.monotonic() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        raw_output = stdout + "\n" + stderr

        # --- OOM detection ---
        if proc.returncode in (-9, 137):
            raise PySCFOOMError(
                f"Process killed (likely OOM), exit code {proc.returncode}"
            )
        if "MemoryError" in raw_output:
            raise PySCFOOMError("PySCF ran out of memory (MemoryError)")

        # --- Parse structured results ---
        results = _parse_results(stdout)
        method = _extract_method(script)
        basis = results.get("basis", "") or _extract_basis(script)
        converged = results.get("converged", True)

        geometry = None
        if "geometry" in results:
            geometry = [
                (a[0], a[1], a[2], a[3]) for a in results["geometry"]
            ]

        dipole = None
        if "dipole" in results:
            d = results["dipole"]
            dipole = (d[0], d[1], d[2])

        result = QChemResult(
            job_id=job_id,
            energy=results.get("energy"),
            converged=converged,
            geometry=geometry,
            frequencies=results.get("frequencies"),
            dipole=dipole,
            wall_time=wall_time,
            engine="pyscf",
            method=method,
            basis=basis,
            raw_output=raw_output,
        )

        # --- Convergence failure ---
        if not converged or "not converge" in raw_output.lower():
            result = result.model_copy(update={"converged": False})
            raise SCFConvergenceError(
                "SCF did not converge", partial_result=result
            )

        # --- Other failures ---
        if proc.returncode != 0:
            raise RuntimeError(
                f"PySCF exited with code {proc.returncode}:\n"
                f"{stderr[:2000]}"
            )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_results(stdout: str) -> dict:
    """Extract the JSON results block emitted by the collector."""
    start = stdout.find(_MARKER_START)
    end = stdout.find(_MARKER_END)
    if start == -1 or end == -1:
        return {}
    blob = stdout[start + len(_MARKER_START) : end].strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


def _extract_method(script: str) -> str:
    """Heuristically extract the computational method from a PySCF script."""
    # DFT functional
    m = re.search(r"\.xc\s*=\s*['\"]([^'\"]+)['\"]", script)
    if m:
        return m.group(1).lower()
    # Post-HF
    if re.search(r"cc\.CCSD\(T\)", script):
        return "ccsd(t)"
    if "cc.CCSD" in script:
        return "ccsd"
    if re.search(r"mp\.(R|U)?MP2", script):
        return "mp2"
    if "ci.CISD" in script:
        return "cisd"
    # HF
    if re.search(r"scf\.(R|U|RO)?HF", script):
        return "hf"
    return ""


def _extract_basis(script: str) -> str:
    """Extract the basis set name from a PySCF script."""
    m = re.search(r"\.basis\s*=\s*['\"]([^'\"]+)['\"]", script)
    return m.group(1) if m else ""
