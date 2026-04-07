"""
test_converter_examples.py
--------------------------
Documented test cases for qchem_converter.py.

Pattern for every test:
    1. Known INPUT  — a real ORCA or Psi4 input string
    2. Run converter
    3. Assert specific EXPECTED OUTPUT properties

Run with:
    pytest test_converter_examples.py -v
"""

import pytest
from qchem_converter import (
    convert, detect_format,
    Psi4Parser, OrcaParser, PySCFGenerator,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FORMAT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatDetection:
    """Auto-detect whether a file is Psi4 or ORCA format."""

    def test_psi4_detected_from_molecule_block(self):
        """Psi4 files always contain 'molecule { ... }'."""
        inp = """
        molecule water {
          0 1
          O  0.000  0.000  0.117
          H  0.000  0.757 -0.469
          H  0.000 -0.757 -0.469
        }
        energy('hf')
        """
        assert detect_format(inp, "calc.inp") == "psi4"

    def test_orca_detected_from_keyword_line(self):
        """ORCA files start with '! keyword ...' lines."""
        inp = """! B3LYP def2-SVP
        * xyz 0 1
        O  0.000  0.000  0.117
        H  0.000  0.757 -0.469
        H  0.000 -0.757 -0.469
        *
        """
        assert detect_format(inp, "calc.inp") == "orca"

    def test_psi4_detected_from_dat_extension(self):
        inp = """
        molecule { 0 1
          H 0 0 0
          H 0 0 0.74
        }
        energy('hf')
        """
        assert detect_format(inp, "calc.dat") == "psi4"

    def test_orca_detected_from_xyz_geometry_block(self):
        inp = "* xyz 0 1\nC 0 0 0\n*"
        assert detect_format(inp, "calc.inp") == "orca"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PSI4 PARSING — geometry extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestPsi4GeometryParsing:
    """Verify atoms, charge, and multiplicity are extracted correctly."""

    PSI4_WATER = """
    molecule water {
      0 1
      O   0.000000   0.000000   0.117176
      H   0.000000   0.757160  -0.468704
      H   0.000000  -0.757160  -0.468704
    }
    set basis 6-31G*
    energy('hf')
    """

    def test_atom_count(self):
        job = Psi4Parser().parse(self.PSI4_WATER)
        assert len(job.atoms) == 3

    def test_atom_symbols(self):
        job = Psi4Parser().parse(self.PSI4_WATER)
        symbols = [a[0] for a in job.atoms]
        assert symbols == ["O", "H", "H"]

    def test_oxygen_coordinates(self):
        """First atom should be O at origin (z = 0.117176)."""
        job = Psi4Parser().parse(self.PSI4_WATER)
        sym, x, y, z = job.atoms[0]
        assert sym == "O"
        assert abs(z - 0.117176) < 1e-6

    def test_neutral_singlet(self):
        job = Psi4Parser().parse(self.PSI4_WATER)
        assert job.charge == 0
        assert job.multiplicity == 1

    def test_charged_molecule(self):
        inp = """
        molecule { -1 1
          F  0.0  0.0  0.0
        }
        energy('hf')
        """
        job = Psi4Parser().parse(inp)
        assert job.charge == -1

    def test_open_shell_multiplicity(self):
        inp = """
        molecule oh { 0 2
          O  0.0  0.0  0.0
          H  0.0  0.0  0.97
        }
        energy('hf')
        """
        job = Psi4Parser().parse(inp)
        assert job.multiplicity == 2
        assert job.unrestricted is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PSI4 PARSING — method and basis set
# ═══════════════════════════════════════════════════════════════════════════════

class TestPsi4MethodBasis:

    def test_b3lyp_method(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nset basis def2-TZVP\nenergy('b3lyp')"
        job = Psi4Parser().parse(inp)
        assert job.method == "b3lyp"

    def test_ccsd_t_method(self):
        """ccsd(t) must be parsed with parentheses intact."""
        inp = "molecule { 0 1\n C 0 0 0\n}\nset basis cc-pVTZ\nenergy('ccsd(t)')"
        job = Psi4Parser().parse(inp)
        assert job.method == "ccsd(t)"

    def test_mp2_method(self):
        inp = "molecule { 0 1\n C 0 0 0\n}\nset basis cc-pVDZ\nenergy('mp2')"
        job = Psi4Parser().parse(inp)
        assert job.method == "mp2"

    def test_basis_normalisation_def2(self):
        """def2-TZVP → def2-tzvp (lowercase canonical form)."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nset basis def2-TZVP\nenergy('hf')"
        job = Psi4Parser().parse(inp)
        assert job.basis == "def2-tzvp"

    def test_basis_normalisation_dunning(self):
        """aug-cc-pVDZ should be preserved exactly."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nset basis aug-cc-pVDZ\nenergy('mp2')"
        job = Psi4Parser().parse(inp)
        assert job.basis == "aug-cc-pvdz"

    def test_basis_normalisation_pople(self):
        inp = "molecule { 0 1\n C 0 0 0\n}\nset basis 6-31G*\nenergy('hf')"
        job = Psi4Parser().parse(inp)
        assert job.basis == "6-31g*"

    def test_maxiter_from_set_block(self):
        inp = """
        molecule { 0 1\n O 0 0 0\n}
        set { basis 6-31G\n maxiter 300 }
        energy('hf')
        """
        job = Psi4Parser().parse(inp)
        assert job.scf_max_cycles == 300


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PSI4 PARSING — job types
# ═══════════════════════════════════════════════════════════════════════════════

class TestPsi4JobTypes:

    def test_single_point_energy(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('b3lyp')"
        assert Psi4Parser().parse(inp).job_type == "energy"

    def test_geometry_optimisation(self):
        inp = "molecule { 0 1\n O 0 0 0\n H 0 0 1\n}\noptimize('b3lyp')"
        assert Psi4Parser().parse(inp).job_type == "opt"

    def test_frequency_calculation(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nfrequencies('hf')"
        assert Psi4Parser().parse(inp).job_type == "freq"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ORCA PARSING
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrcaParsing:

    ORCA_WATER = """! B3LYP def2-TZVP TightSCF Opt

    %pal nprocs 4 end
    %maxcore 2000

    %scf MaxIter 250 end

    * xyz 0 1
    O   0.000000   0.000000   0.117176
    H   0.000000   0.757160  -0.468704
    H   0.000000  -0.757160  -0.468704
    *
    """

    def test_method_detected(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.method == "b3lyp"

    def test_basis_detected(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.basis == "def2-tzvp"

    def test_opt_job_type(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.job_type == "opt"

    def test_nprocs(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.nprocs == 4

    def test_memory(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.memory_mb == 2000

    def test_geometry_atoms(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert len(job.atoms) == 3
        assert job.atoms[0][0] == "O"

    def test_charge_and_multiplicity(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.charge == 0
        assert job.multiplicity == 1

    def test_scf_maxiter(self):
        job = OrcaParser().parse(self.ORCA_WATER)
        assert job.scf_max_cycles == 250

    def test_open_shell_uhf(self):
        inp = "! UHF cc-pVDZ\n* xyz 0 2\nO 0 0 0\nH 0 0 1\n*"
        job = OrcaParser().parse(inp)
        assert job.unrestricted is True
        assert job.multiplicity == 2

    def test_freq_job_type(self):
        inp = "! B3LYP def2-SVP Freq\n* xyz 0 1\nO 0 0 0\n*"
        job = OrcaParser().parse(inp)
        assert job.job_type == "freq"

    def test_cpcm_solvent(self):
        inp = "! PBE0 def2-SVP CPCM(water)\n%cpcm epsilon 80.0 end\n* xyz 0 1\nO 0 0 0\n*"
        job = OrcaParser().parse(inp)
        assert job.solvent_model == "ddcosmo"
        assert abs(job.solvent_eps - 80.0) < 1e-9

    def test_ccsd_t(self):
        inp = "! CCSD(T) cc-pVTZ\n* xyz 0 1\nC 0 0 0\n*"
        job = OrcaParser().parse(inp)
        assert job.method == "ccsd(t)"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DOUBLE-HYBRID FUNCTIONALS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.xfail(
    reason="Double-hybrid DFT not yet in qchem_converter.py — "
           "implement QChemJob.double_hybrid fields and DOUBLE_HYBRID_MAP. "
           "These tests define the target behaviour.",
    strict=True,
)
class TestDoubleHybrid:

    def test_b2plyp_detected_psi4(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nset basis def2-TZVP\nenergy('b2plyp')"
        job = Psi4Parser().parse(inp)
        assert job.double_hybrid is True
        assert job.method == "b2plyp"

    def test_b2plyp_ax_coefficient(self):
        """B2PLYP has 53% HF exchange (published value)."""
        job = Psi4Parser().parse(
            "molecule { 0 1\n O 0 0 0\n}\nenergy('b2plyp')"
        )
        assert abs(job.dh_ax - 0.53) < 1e-9

    def test_b2plyp_ac_coefficient(self):
        """B2PLYP has 27% MP2 correlation (published value)."""
        job = Psi4Parser().parse(
            "molecule { 0 1\n O 0 0 0\n}\nenergy('b2plyp')"
        )
        assert abs(job.dh_ac - 0.27) < 1e-9

    def test_b2gp_plyp_coefficients(self):
        """B2GP-PLYP: ax=0.65, ac=0.36 (Karton et al. 2008)."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('b2gp-plyp')"
        job = Psi4Parser().parse(inp)
        assert abs(job.dh_ax - 0.65) < 1e-9
        assert abs(job.dh_ac - 0.36) < 1e-9

    def test_pwpb95_sos_type(self):
        """PWPB95 is SOS (same-spin scale = 0, only opposite-spin)."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('pwpb95')"
        job = Psi4Parser().parse(inp)
        assert job.dh_ss_scale == 0.0
        assert abs(job.dh_os_scale - 0.73) < 1e-9

    def test_dsd_pbep86_sos_type(self):
        """DSD-PBEP86 is also SOS."""
        inp = "! DSD-PBEP86 def2-TZVP\n* xyz 0 1\nO 0 0 0\n*"
        job = OrcaParser().parse(inp)
        assert job.double_hybrid is True
        assert job.dh_ss_scale == 0.0

    def test_b2plyp_orca_detected(self):
        inp = "! B2PLYP def2-TZVP\n* xyz 0 1\nO 0 0 0\n*"
        job = OrcaParser().parse(inp)
        assert job.double_hybrid is True
        assert abs(job.dh_ax - 0.53) < 1e-9

    def test_pbe0_dh_coefficients(self):
        """PBE0-DH: ax=0.50, ac=0.125 — verify via generated xc string."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('pbe0-dh')"
        job = Psi4Parser().parse(inp)
        # If recognised as double hybrid, ax and ac must match published values
        if job.double_hybrid:
            assert abs(job.dh_ax - 0.50) < 1e-9
            assert abs(job.dh_ac - 0.125) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PYSCF CODE GENERATION — method dispatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeGenMethod:
    """Verify the generated PySCF script uses the correct classes."""

    def test_hf_generates_rhf(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('hf')"
        out = convert(inp, fmt="psi4")
        assert "scf.RHF" in out

    def test_open_shell_generates_uhf(self):
        inp = "molecule { 0 2\n O 0 0 0\n}\nset reference uhf\nenergy('hf')"
        out = convert(inp, fmt="psi4")
        assert "scf.UHF" in out

    def test_dft_generates_rks(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('b3lyp')"
        out = convert(inp, fmt="psi4")
        assert "dft.RKS" in out

    def test_dft_open_shell_generates_uks(self):
        inp = "molecule { 0 2\n O 0 0 0\n}\nset reference uhf\nenergy('b3lyp')"
        out = convert(inp, fmt="psi4")
        assert "dft.UKS" in out

    def test_dft_xc_string_set(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('pbe0')"
        out = convert(inp, fmt="psi4")
        assert "mf.xc = 'pbe0'" in out

    def test_mp2_imports_mp_module(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('mp2')"
        out = convert(inp, fmt="psi4")
        assert "from pyscf import mp" in out

    def test_ccsd_imports_cc_module(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('ccsd')"
        out = convert(inp, fmt="psi4")
        assert "from pyscf import cc" in out

    def test_ccsd_t_includes_triples(self):
        inp = "molecule { 0 1\n C 0 0 0\n}\nenergy('ccsd(t)')"
        out = convert(inp, fmt="psi4")
        assert "ccsd_t" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PYSCF CODE GENERATION — molecule block
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeGenMolecule:

    def test_charge_written(self):
        inp = "molecule { -1 1\n F 0 0 0\n}\nenergy('hf')"
        out = convert(inp, fmt="psi4")
        assert "mol.charge = -1" in out

    def test_spin_written(self):
        """spin = multiplicity - 1."""
        inp = "molecule { 0 2\n O 0 0 0\n H 0 0 1\n}\nenergy('hf')"
        out = convert(inp, fmt="psi4")
        assert "mol.spin   = 1" in out

    def test_basis_written(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nset basis def2-TZVP\nenergy('hf')"
        out = convert(inp, fmt="psi4")
        assert "mol.basis  = 'def2-tzvp'" in out

    def test_oxygen_atom_in_output(self):
        inp = "molecule { 0 1\n O 0.0 0.0 0.117\n H 0.0 0.757 -0.469\n}\nenergy('hf')"
        out = convert(inp, fmt="psi4")
        assert "O" in out
        assert "0.11700000" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PYSCF CODE GENERATION — job types
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeGenJobTypes:

    def test_opt_calls_optimize(self):
        inp = "molecule { 0 1\n O 0 0 0\n H 0 0 1\n}\noptimize('b3lyp')"
        out = convert(inp, fmt="psi4")
        assert "optimize(mf)" in out

    def test_freq_calls_hessian(self):
        inp = "molecule { 0 1\n O 0 0 0\n H 0 0 1\n}\nfrequencies('hf')"
        out = convert(inp, fmt="psi4")
        assert "Hessian" in out

    def test_freq_imports_thermo(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nfrequencies('hf')"
        out = convert(inp, fmt="psi4")
        assert "thermo" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PYSCF CODE GENERATION — double hybrid output
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.xfail(
    reason="Double-hybrid code generation not yet implemented in converter.",
    strict=True,
)
class TestCodeGenDoubleHybrid:

    def test_b2plyp_xc_string_contains_hf_fraction(self):
        """The generated xc string must encode 0.53 HF exchange explicitly."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nset basis def2-TZVP\nenergy('b2plyp')"
        out = convert(inp, fmt="psi4")
        assert "0.5300*HF" in out

    def test_b2plyp_imports_mp(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('b2plyp')"
        out = convert(inp, fmt="psi4")
        assert "from pyscf import mp" in out

    def test_b2plyp_pt2_correction_present(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('b2plyp')"
        out = convert(inp, fmt="psi4")
        assert "pt2 = mp.MP2" in out
        assert "ac = 0.27" in out

    def test_pwpb95_ss_scale_zero(self):
        """PWPB95 SOS: same-spin PT2 contribution must be zero."""
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('pwpb95')"
        out = convert(inp, fmt="psi4")
        assert "pt2_ss_scale = 0.0" in out

    def test_pwpb95_os_scale(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('pwpb95')"
        out = convert(inp, fmt="psi4")
        assert "pt2_os_scale = 0.73" in out

    def test_dh_energy_printed(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nenergy('b2plyp')"
        out = convert(inp, fmt="psi4")
        assert "E(DH-total)" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SOLVENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolvent:

    def test_orca_cpcm_wraps_ddcosmo(self):
        inp = "! PBE0 def2-SVP CPCM(water)\n%cpcm epsilon 80.0 end\n* xyz 0 1\nO 0 0 0\n*"
        out = convert(inp, fmt="orca")
        assert "ddCOSMO" in out

    def test_orca_cpcm_sets_eps(self):
        inp = "! PBE0 def2-SVP\n%cpcm epsilon 78.4 end\n* xyz 0 1\nO 0 0 0\n*"
        out = convert(inp, fmt="orca")
        assert "eps = 78.4" in out

    def test_psi4_pcm_wraps_ddcosmo(self):
        inp = "molecule { 0 1\n O 0 0 0\n}\nset pcm {eps = 78.4}\nenergy('b3lyp')"
        out = convert(inp, fmt="psi4")
        assert "ddCOSMO" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 12. RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

class TestResources:

    def test_orca_nprocs_in_output(self):
        inp = "! HF STO-3G\n%pal nprocs 8 end\n* xyz 0 1\nH 0 0 0\n*"
        out = convert(inp, fmt="orca")
        assert "lib.num_threads(8)" in out

    def test_orca_memory_in_output(self):
        inp = "! HF STO-3G\n%maxcore 4096\n* xyz 0 1\nH 0 0 0\n*"
        out = convert(inp, fmt="orca")
        assert "lib.max_memory = 4096" in out

    def test_psi4_memory_gb_converted_to_mb(self):
        inp = "memory 4 GB\nmolecule { 0 1\n O 0 0 0\n}\nenergy('hf')"
        job = Psi4Parser().parse(inp)
        assert job.memory_mb == 4096


# ═══════════════════════════════════════════════════════════════════════════════
# 13. END-TO-END — full ORCA → PySCF round trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_orca_b3lyp_opt_round_trip(self):
        """Full ORCA B3LYP optimisation → valid PySCF script."""
        orca_inp = """! B3LYP def2-TZVP TightSCF Opt
        %pal nprocs 4 end
        %maxcore 2000
        * xyz 0 1
        O   0.000000   0.000000   0.117176
        H   0.000000   0.757160  -0.468704
        H   0.000000  -0.757160  -0.468704
        *
        """
        out = convert(orca_inp, fmt="orca", source_name="water.inp")
        # Must be valid Python
        import ast
        ast.parse(out)                            # raises SyntaxError if broken
        # Must contain expected elements
        assert "gto.Mole" in out
        assert "dft.RKS" in out
        assert "mf.xc = 'b3lyp'" in out
        assert "mol.basis  = 'def2-tzvp'" in out
        assert "optimize(mf)" in out
        assert "lib.num_threads(4)" in out

    def test_psi4_ccsd_t_round_trip(self):
        """Full Psi4 CCSD(T) → valid PySCF script with triples correction."""
        psi4_inp = """
        molecule methane {
          0 1
          C   0.000000   0.000000   0.000000
          H   0.629118   0.629118   0.629118
          H  -0.629118  -0.629118   0.629118
          H  -0.629118   0.629118  -0.629118
          H   0.629118  -0.629118  -0.629118
        }
        set basis cc-pVTZ
        energy('ccsd(t)')
        """
        out = convert(psi4_inp, fmt="psi4", source_name="methane.inp")
        import ast
        ast.parse(out)
        assert "from pyscf import cc" in out
        assert "ccsd_t" in out
        assert "mol.basis  = 'cc-pvtz'" in out
        assert "mol.atom" in out   # geometry block present

    @pytest.mark.xfail(
        reason="DSD-PBEP86 double-hybrid not yet supported.",
        strict=True,
    )
    def test_orca_dsd_pbep86_round_trip(self):
        """DSD-PBEP86 (SOS double hybrid) → correct PT2 scaling."""
        orca_inp = """! DSD-PBEP86 def2-TZVP TightSCF
        * xyz 0 1
        C   0.000000   0.000000   0.000000
        O   0.000000   0.000000   1.200000
        *
        """
        out = convert(orca_inp, fmt="orca", source_name="co.inp")
        import ast
        ast.parse(out)
        assert "pt2_ss_scale = 0.0" in out    # SOS: no same-spin
        assert "mp.MP2" in out
        assert "dft.RKS" in out

    def test_auto_detect_and_convert_psi4(self):
        """Auto-detect should handle a Psi4 file end-to-end."""
        psi4_inp = """
        molecule water { 0 1
          O 0 0 0.117
          H 0 0.757 -0.469
          H 0 -0.757 -0.469
        }
        set basis 6-31G*
        energy('b3lyp')
        """
        out = convert(psi4_inp, fmt="auto", source_name="test.dat")
        assert "gto.Mole" in out
        assert "b3lyp" in out

    def test_generated_script_is_valid_python(self):
        """Every generated script must parse as valid Python."""
        import ast
        cases = [
            ("molecule { 0 1\n O 0 0 0\n}\nenergy('hf')", "psi4"),
            ("molecule { 0 1\n O 0 0 0\n}\noptimize('b3lyp')", "psi4"),
            ("molecule { 0 1\n O 0 0 0\n}\nfrequencies('mp2')", "psi4"),
            ("! B3LYP def2-SVP\n* xyz 0 1\nO 0 0 0\n*", "orca"),
            ("! CCSD(T) cc-pVDZ\n* xyz 0 1\nO 0 0 0\n*", "orca"),
            ("! B2PLYP def2-TZVP\n* xyz 0 1\nO 0 0 0\n*", "orca"),
        ]
        for inp, fmt in cases:
            out = convert(inp, fmt=fmt)
            try:
                ast.parse(out)
            except SyntaxError as e:
                pytest.fail(f"Generated script is invalid Python for {fmt!r}: {e}")
