"""Tests for tools/run_pyscf.py — no PySCF installation required."""

import textwrap

import pytest

from tools.run_pyscf import (
    PySCFOOMError,
    PySCFTimeoutError,
    SCFConvergenceError,
    _extract_basis,
    _extract_method,
    _parse_results,
    run_pyscf,
)


# ------------------------------------------------------------------ helpers
class TestParseResults:
    def test_valid_json(self):
        stdout = (
            "some output\n"
            "<<<PYSCF_RESULTS>>>\n"
            '{"energy": -76.0, "converged": true}\n'
            "<<<END_PYSCF_RESULTS>>>\n"
        )
        assert _parse_results(stdout) == {"energy": -76.0, "converged": True}

    def test_no_markers(self):
        assert _parse_results("just text") == {}

    def test_malformed_json(self):
        stdout = "<<<PYSCF_RESULTS>>>\n{bad\n<<<END_PYSCF_RESULTS>>>\n"
        assert _parse_results(stdout) == {}

    def test_empty_between_markers(self):
        stdout = "<<<PYSCF_RESULTS>>>\n\n<<<END_PYSCF_RESULTS>>>\n"
        assert _parse_results(stdout) == {}


class TestExtractMethod:
    def test_dft(self):
        assert _extract_method("mf.xc = 'B3LYP'") == "b3lyp"

    def test_hf(self):
        assert _extract_method("mf = scf.RHF(mol)") == "hf"

    def test_uhf(self):
        assert _extract_method("mf = scf.UHF(mol)") == "hf"

    def test_ccsd(self):
        assert _extract_method("mycc = cc.CCSD(mf)") == "ccsd"

    def test_ccsd_t(self):
        assert _extract_method("e = cc.CCSD(T)(mf).run()") == "ccsd(t)"

    def test_mp2(self):
        assert _extract_method("pt = mp.MP2(mf)") == "mp2"

    def test_ump2(self):
        assert _extract_method("pt = mp.UMP2(mf)") == "mp2"

    def test_cisd(self):
        assert _extract_method("myci = ci.CISD(mf)") == "cisd"

    def test_unknown(self):
        assert _extract_method("print('hello')") == ""


class TestExtractBasis:
    def test_found(self):
        assert _extract_basis("mol.basis = 'cc-pvdz'") == "cc-pvdz"

    def test_double_quotes(self):
        assert _extract_basis('mol.basis = "def2-tzvp"') == "def2-tzvp"

    def test_not_found(self):
        assert _extract_basis("no basis here") == ""


# --------------------------------------------------------- run_pyscf tests

def _mock_mf(energy: float = -76.026760, converged: bool = True) -> str:
    """Return a script that defines a mock SCF object."""
    return textwrap.dedent(f"""\
        class _MF:
            e_tot = {energy}
            converged = {converged}
        mf = _MF()
    """)


def _mock_mol(basis: str = "cc-pvdz") -> str:
    """Return a script that defines a mock Mol object with geometry."""
    return textwrap.dedent(f"""\
        class _Mol:
            basis = '{basis}'
            natm = 3
            def atom_coords(self, unit='ANG'):
                return [
                    [0.0000, 0.0000,  0.1173],
                    [0.0000, 0.7572, -0.4692],
                    [0.0000, -0.7572, -0.4692],
                ]
            def atom_symbol(self, i):
                return ['O', 'H', 'H'][i]
        mol = _Mol()
    """)


class TestRunPySCFSuccess:
    def test_basic_energy(self):
        script = _mock_mf() + "\nmf.xc = 'b3lyp'\n" + _mock_mol()
        result = run_pyscf(script, timeout=30)

        assert result.engine == "pyscf"
        assert result.converged is True
        assert result.energy == pytest.approx(-76.026760)
        assert result.method == "b3lyp"
        assert result.basis == "cc-pvdz"
        assert result.wall_time > 0
        assert result.job_id  # non-empty

    def test_geometry_extraction(self):
        script = _mock_mf() + _mock_mol("sto-3g")
        result = run_pyscf(script, timeout=30)

        assert result.geometry is not None
        assert len(result.geometry) == 3
        assert result.geometry[0][0] == "O"
        assert result.geometry[1][0] == "H"
        assert result.geometry[0][1] == pytest.approx(0.0)

    def test_dipole_extraction(self):
        script = _mock_mf() + "\ndip = [0.0, 0.0, 1.857]\n"
        result = run_pyscf(script, timeout=30)

        assert result.dipole is not None
        assert result.dipole[2] == pytest.approx(1.857)

    def test_frequencies_extraction(self):
        script = _mock_mf() + "\nfreq = [1648.5, 3832.1, 3942.7]\n"
        result = run_pyscf(script, timeout=30)

        assert result.frequencies is not None
        assert len(result.frequencies) == 3
        assert result.frequencies[0] == pytest.approx(1648.5)

    def test_raw_output_captured(self):
        script = "print('hello pyscf')\n" + _mock_mf()
        result = run_pyscf(script, timeout=30)
        assert "hello pyscf" in result.raw_output

    def test_basis_from_script_fallback(self):
        """When the collector can't read mol.basis, fall back to regex."""
        script = _mock_mf() + "\nmol = None\n"
        script_with_basis = "mol_obj = None\nmol_obj.basis = 'aug-cc-pvtz'\n" + script
        # mol_obj.basis line won't help collector (wrong name), but regex finds it
        result = run_pyscf(
            _mock_mf() + "\n# mol.basis = '6-31g'\n", timeout=30
        )
        # regex picks up the commented-out basis (it's a heuristic)
        assert result.basis == "6-31g"


class TestRunPySCFConvergence:
    def test_converged_false_in_result(self):
        script = _mock_mf(energy=-75.5, converged=False)
        with pytest.raises(SCFConvergenceError) as exc_info:
            run_pyscf(script, timeout=30)

        err = exc_info.value
        assert err.partial_result is not None
        assert err.partial_result.energy == pytest.approx(-75.5)
        assert err.partial_result.converged is False

    def test_convergence_warning_in_stdout(self):
        script = 'print("warn: SCF not converged")\n' + _mock_mf()
        with pytest.raises(SCFConvergenceError):
            run_pyscf(script, timeout=30)


class TestRunPySCFTimeout:
    def test_timeout_raises(self):
        script = "import time; time.sleep(10)"
        with pytest.raises(PySCFTimeoutError, match="timed out"):
            run_pyscf(script, timeout=1)


class TestRunPySCFOOM:
    def test_memory_error_in_output(self):
        script = textwrap.dedent("""\
            import sys
            sys.stderr.write("MemoryError\\n")
            sys.exit(1)
        """)
        with pytest.raises(PySCFOOMError, match="MemoryError"):
            run_pyscf(script, timeout=30)

    def test_exit_code_137(self):
        script = "import sys; sys.exit(137)"
        with pytest.raises(PySCFOOMError, match="137"):
            run_pyscf(script, timeout=30)


class TestRunPySCFRuntimeError:
    def test_script_exception(self):
        script = "raise ValueError('something broke')"
        with pytest.raises(RuntimeError, match="exited with code"):
            run_pyscf(script, timeout=30)
