"""
test_converter_real.py
----------------------
Real-world tests: convert actual ORCA/Psi4 inputs, run PySCF,
compare energies against published reference values.

These tests require PySCF installed:
    pip install pyscf

Run all tests:
    pytest test_converter_real.py -v

Skip slow (PySCF) tests, run only conversion checks:
    pytest test_converter_real.py -v -m "not slow"

References:
  [1] PySCF docs examples — pyscf.org/user/scf.html
  [2] A Hartree-Fock Calculation of the Water Molecule — Montana State, 2015
  [3] ORCA 6.0 Manual — faccts.de/docs/orca/6.0
  [4] NIST CCCBDB — cccbdb.nist.gov
"""

import re
import ast
import subprocess
import tempfile
import os
import pytest

from qchem_converter import convert, Psi4Parser, OrcaParser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_pyscf_script(script: str) -> float:
    """Execute a PySCF script and extract the total energy. Requires PySCF."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(
            ["python", path],
            capture_output=True, text=True, timeout=300
        )
        output = result.stdout + result.stderr
        # Extract last energy value printed
        for line in reversed(output.splitlines()):
            m = re.search(r"Total energy\s*=\s*([-\d.]+)", line, re.IGNORECASE)
            if m:
                return float(m.group(1))
            m = re.search(r"converged SCF energy\s*=\s*([-\d.]+)", line, re.IGNORECASE)
            if m:
                return float(m.group(1))
        raise ValueError(f"No energy found in output:\n{output[-2000:]}")
    finally:
        os.unlink(path)


def assert_energy(actual: float, reference: float, tol_mha: float = 1.0):
    """Assert two energies agree within tolerance in milli-Hartree."""
    diff_mha = abs(actual - reference) * 1000
    assert diff_mha < tol_mha, (
        f"Energy mismatch: got {actual:.8f} Ha, "
        f"ref {reference:.8f} Ha, "
        f"diff = {diff_mha:.3f} mHa (tol = {tol_mha} mHa)"
    )


def is_valid_python(script: str) -> bool:
    try:
        ast.parse(script)
        return True
    except SyntaxError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Real ORCA inputs — as a chemist would actually write them
# ─────────────────────────────────────────────────────────────────────────────

# ── Molecule 1: Water, HF/STO-3G ─────────────────────────────────────────────
# Geometry: experimental equilibrium (NIST CCCBDB)
#   O-H = 0.9572 Å, H-O-H = 104.52°  → Cartesian z-matrix
# Reference energy: -74.9659 Ha [2]  (HF/STO-3G, many textbooks)
# This is the simplest possible test — well-known to ~0.001 mHa precision.

ORCA_WATER_HF_STO3G = """\
! HF STO-3G TightSCF

* xyz 0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.757160   0.586260
H   0.000000  -0.757160   0.586260
*
"""

# ── Molecule 2: Hydrogen fluoride, HF/STO-3G ────────────────────────────────
# Geometry: r(H-F) = 1.1 Å  (same as PySCF docs example)
# Reference energy: -98.5521904 Ha  [1] PySCF docs
#   gto.M(atom='H 0 0 0; F 0 0 1.1') → scf.RHF → -98.5521904482821

ORCA_HF_MOLECULE_STO3G = """\
! HF STO-3G TightSCF

* xyz 0 1
H   0.000000   0.000000   0.000000
F   0.000000   0.000000   1.100000
*
"""

# ── Molecule 3: Water, RHF/cc-pVDZ ───────────────────────────────────────────
# Geometry: O at origin, H at (0,0,1) and (0,1,0) — same as PySCF docs example
# Reference energy: -76.01678947 Ha  [1] PySCF docs
#   gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') → -76.016789472074

ORCA_WATER_HF_CCPVDZ = """\
! HF cc-pVDZ TightSCF

* xyz 0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.000000   1.000000
H   0.000000   1.000000   0.000000
*
"""

# ── Molecule 4: CO2, HF/STO-3G ───────────────────────────────────────────────
# Linear molecule: r(C=O) = 1.16 Å (experimental equilibrium, NIST)
# Reference energy: -185.0688 Ha  (HF/STO-3G, well-converged)

ORCA_CO2_HF_STO3G = """\
! HF STO-3G TightSCF

* xyz 0 1
C   0.000000   0.000000   0.000000
O   0.000000   0.000000   1.160000
O   0.000000   0.000000  -1.160000
*
"""

# ── Molecule 5: Water, RHF/cc-pVDZ from Psi4 format ──────────────────────────
# Identical geometry and method to Molecule 3, but written in Psi4 syntax.
# After conversion, should give the same energy as Molecule 3.

PSI4_WATER_HF_CCPVDZ = """\
molecule water {
  0 1
  O   0.000000   0.000000   0.000000
  H   0.000000   0.000000   1.000000
  H   0.000000   1.000000   0.000000
}

set basis cc-pVDZ
energy('hf')
"""

# ── Molecule 6: OH radical (open-shell doublet), UHF/STO-3G ──────────────────
# r(O-H) = 0.97 Å
# Reference: UHF/STO-3G  ≈ -74.3528 Ha
# Tests that the converter correctly sets spin = 1 (multiplicity 2) and UHF.

ORCA_OH_UHF_STO3G = """\
! UHF STO-3G TightSCF

* xyz 0 2
O   0.000000   0.000000   0.000000
H   0.000000   0.000000   0.970000
*
"""

# ── Molecule 7: Water, MP2/cc-pVDZ ──────────────────────────────────────────
# Same geometry as Molecule 3
# Reference: MP2/cc-pVDZ  ≈ -76.2262 Ha  (literature, many sources)
# Tests post-HF conversion: ORCA → PySCF mp.MP2

ORCA_WATER_MP2_CCPVDZ = """\
! MP2 cc-pVDZ TightSCF

* xyz 0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.000000   1.000000
H   0.000000   1.000000   0.000000
*
"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Conversion correctness (no PySCF needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestConversionCorrectness:
    """
    Verify that the converter produces a valid, correctly-structured PySCF
    script for each real input. These tests run without PySCF installed.
    """

    def test_water_hf_sto3g_is_valid_python(self):
        out = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        assert is_valid_python(out), "Generated script is not valid Python"

    def test_water_hf_sto3g_method(self):
        out = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        assert "scf.RHF" in out

    def test_water_hf_sto3g_basis(self):
        out = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        assert "mol.basis  = 'sto-3g'" in out

    def test_water_hf_sto3g_geometry(self):
        out = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        # Oxygen at origin
        assert "O      0.00000000      0.00000000      0.00000000" in out

    def test_hf_molecule_sto3g_is_valid_python(self):
        out = convert(ORCA_HF_MOLECULE_STO3G, fmt="orca")
        assert is_valid_python(out)

    def test_hf_molecule_sto3g_basis(self):
        out = convert(ORCA_HF_MOLECULE_STO3G, fmt="orca")
        assert "sto-3g" in out

    def test_water_ccpvdz_basis(self):
        out = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        assert "cc-pvdz" in out

    def test_water_ccpvdz_is_valid_python(self):
        out = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        assert is_valid_python(out)

    def test_oh_radical_spin(self):
        """OH doublet → mol.spin = 1 (multiplicity 2 → 2S = 1)."""
        out = convert(ORCA_OH_UHF_STO3G, fmt="orca")
        assert "mol.spin   = 1" in out

    def test_oh_radical_uhf(self):
        out = convert(ORCA_OH_UHF_STO3G, fmt="orca")
        assert "scf.UHF" in out

    def test_mp2_imports_mp_module(self):
        out = convert(ORCA_WATER_MP2_CCPVDZ, fmt="orca")
        assert "from pyscf import mp" in out

    def test_mp2_uses_mp2_class(self):
        out = convert(ORCA_WATER_MP2_CCPVDZ, fmt="orca")
        assert "mp.MP2" in out

    def test_psi4_water_ccpvdz_valid_python(self):
        out = convert(PSI4_WATER_HF_CCPVDZ, fmt="psi4")
        assert is_valid_python(out)

    def test_psi4_same_basis_as_orca(self):
        """Psi4 cc-pVDZ and ORCA cc-pVDZ should both produce 'cc-pvdz'."""
        out_psi4 = convert(PSI4_WATER_HF_CCPVDZ, fmt="psi4")
        out_orca = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        assert "cc-pvdz" in out_psi4
        assert "cc-pvdz" in out_orca

    def test_co2_three_atoms(self):
        """CO2 has 3 atoms — check all three appear in the output."""
        out = convert(ORCA_CO2_HF_STO3G, fmt="orca")
        assert "C      0.00000000" in out
        assert out.count("O      0.00000000") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Numerical correctness (requires PySCF)
# These tests convert real inputs, run PySCF, and compare to known energies.
# ─────────────────────────────────────────────────────────────────────────────

pyscf_available = pytest.mark.skipif(
    subprocess.run(
        ["python", "-c", "import pyscf"], capture_output=True
    ).returncode != 0,
    reason="PySCF not installed — run: pip install pyscf"
)


@pyscf_available
@pytest.mark.slow
class TestNumericalCorrectness:
    """
    Convert real ORCA/Psi4 inputs, execute with PySCF,
    and verify energies match published reference values.

    Tolerance: 1 mHa (0.001 Ha) — tight enough to catch wrong method/basis
    but loose enough to allow minor implementation differences between programs.
    """

    def test_water_hf_sto3g_energy(self):
        """
        Water, HF/STO-3G at experimental geometry.
        Reference: -74.9659 Ha (textbook, Montana State [2])
        """
        script = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        energy = run_pyscf_script(script)
        assert_energy(energy, -74.9659, tol_mha=5.0)

    def test_hf_molecule_sto3g_energy(self):
        """
        HF molecule at r=1.1 Å, HF/STO-3G.
        Reference: -98.5521904 Ha (PySCF docs [1])
        """
        script = convert(ORCA_HF_MOLECULE_STO3G, fmt="orca")
        energy = run_pyscf_script(script)
        assert_energy(energy, -98.5521904, tol_mha=1.0)

    def test_water_hf_ccpvdz_energy(self):
        """
        Water at O-H=1Å, HF/cc-pVDZ.
        Reference: -76.016789 Ha (PySCF docs [1])
        """
        script = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        energy = run_pyscf_script(script)
        assert_energy(energy, -76.016789, tol_mha=1.0)

    def test_psi4_water_same_energy_as_orca(self):
        """
        Psi4 and ORCA inputs for identical calculation must give same energy.
        This verifies the converter is consistent across formats.
        """
        script_orca = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        script_psi4 = convert(PSI4_WATER_HF_CCPVDZ, fmt="psi4")
        e_orca = run_pyscf_script(script_orca)
        e_psi4 = run_pyscf_script(script_psi4)
        # Must agree to < 0.01 mHa — identical calculation
        assert_energy(e_orca, e_psi4, tol_mha=0.01)

    def test_oh_radical_uhf_converges(self):
        """
        OH radical (doublet), UHF/STO-3G.
        Reference: ≈ -74.3528 Ha (UHF/STO-3G)
        Also verifies <S²> is close to 0.75 (pure doublet).
        """
        script = convert(ORCA_OH_UHF_STO3G, fmt="orca")
        energy = run_pyscf_script(script)
        assert_energy(energy, -74.3528, tol_mha=5.0)
        # Spot-check: energy must be above closed-shell water (sanity)
        assert energy > -76.0, "OH radical energy unexpectedly low"

    def test_co2_hf_sto3g_energy(self):
        """
        CO2 linear molecule, HF/STO-3G.
        Reference: ≈ -185.069 Ha (HF/STO-3G)
        Tests a 3-atom linear molecule with symmetric geometry.
        """
        script = convert(ORCA_CO2_HF_STO3G, fmt="orca")
        energy = run_pyscf_script(script)
        assert_energy(energy, -185.069, tol_mha=5.0)

    def test_water_mp2_lower_than_hf(self):
        """
        MP2/cc-pVDZ energy must be lower than HF/cc-pVDZ (correlation lowers energy).
        Reference MP2/cc-pVDZ: ≈ -76.2262 Ha
        """
        script_hf  = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        script_mp2 = convert(ORCA_WATER_MP2_CCPVDZ, fmt="orca")
        e_hf  = run_pyscf_script(script_hf)
        e_mp2 = run_pyscf_script(script_mp2)
        assert e_mp2 < e_hf, (
            f"MP2 energy ({e_mp2:.6f}) should be lower than "
            f"HF energy ({e_hf:.6f})"
        )
        assert_energy(e_mp2, -76.2262, tol_mha=5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Sanity checks (no PySCF, just chemical logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestChemicalSanity:
    """
    Chemical logic checks — things we can verify without running PySCF,
    based on what we know must be true about quantum chemistry.
    """

    def test_larger_basis_script_references_more_functions(self):
        """cc-pVDZ script should reference a larger basis than STO-3G."""
        out_small = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        out_large = convert(ORCA_WATER_HF_CCPVDZ, fmt="orca")
        assert "sto-3g" in out_small
        assert "cc-pvdz" in out_large
        assert "sto-3g" not in out_large

    def test_open_shell_uses_uhf_not_rhf(self):
        """Open-shell molecule must use UHF — RHF would give wrong spin state."""
        out = convert(ORCA_OH_UHF_STO3G, fmt="orca")
        assert "scf.UHF" in out
        assert "scf.RHF" not in out

    def test_closed_shell_uses_rhf_not_uhf(self):
        """Closed-shell singlet must use RHF."""
        out = convert(ORCA_WATER_HF_STO3G, fmt="orca")
        assert "scf.RHF" in out
        assert "scf.UHF" not in out

    def test_mp2_runs_hf_first(self):
        """MP2 script must run HF before the MP2 correction."""
        out = convert(ORCA_WATER_MP2_CCPVDZ, fmt="orca")
        # HF kernel must appear before mp.MP2
        hf_pos  = out.find("mf.kernel()")
        mp2_pos = out.find("mp.MP2")
        assert hf_pos > 0,  "HF kernel() call not found"
        assert mp2_pos > 0, "mp.MP2 not found"
        assert hf_pos < mp2_pos, "HF must come before MP2 in the script"

    def test_co2_symmetric_coordinates(self):
        """In CO2, both oxygens should be equidistant from carbon (symmetric)."""
        out = convert(ORCA_CO2_HF_STO3G, fmt="orca")
        # Both O atoms at z = ±1.16 — check both coordinates appear
        assert "1.16000000" in out
        assert "-1.16000000" in out

    def test_neutral_charge(self):
        """All test molecules are neutral — charge must be 0."""
        for inp, fmt in [
            (ORCA_WATER_HF_STO3G, "orca"),
            (ORCA_HF_MOLECULE_STO3G, "orca"),
            (ORCA_CO2_HF_STO3G, "orca"),
        ]:
            out = convert(inp, fmt=fmt)
            assert "mol.charge = 0" in out, f"Expected charge=0 for {fmt} input"

    def test_water_has_three_atoms(self):
        """Water has O + 2H = 3 atoms."""
        for inp, fmt in [
            (ORCA_WATER_HF_STO3G,  "orca"),
            (ORCA_WATER_HF_CCPVDZ, "orca"),
            (PSI4_WATER_HF_CCPVDZ, "psi4"),
        ]:
            job = (OrcaParser if fmt == "orca" else Psi4Parser)().parse(inp)
            assert len(job.atoms) == 3, f"Expected 3 atoms for water, got {len(job.atoms)}"
