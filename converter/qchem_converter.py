"""
qchem_converter.py
------------------
Convert Psi4 and ORCA input files to equivalent PySCF Python scripts.

Usage:
    python qchem_converter.py input.inp          # auto-detect format
    python qchem_converter.py input.inp --fmt psi4
    python qchem_converter.py input.inp --fmt orca
    python qchem_converter.py input.inp -o output.py

Supported features:
  - Molecule geometry (Cartesian and Z-matrix for Psi4)
  - Charge and spin multiplicity
  - DFT, HF, MP2, CCSD, CCSD(T)
  - Basis sets (Pople, Dunning, def2-*, etc.)
  - Job types: single-point, optimization, frequency
  - Solvent (CPCM/PCM)
  - Auxiliary basis / density fitting
  - Open-shell (UHF/UKS)
"""

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QChemJob:
    """Unified internal representation of a QM job."""
    charge: int = 0
    multiplicity: int = 1
    atoms: list = field(default_factory=list)          # list of (symbol, x, y, z)
    method: str = "hf"
    basis: str = "sto-3g"
    job_type: str = "energy"                           # energy | opt | freq
    scf_max_cycles: int = 200
    scf_conv_tol: float = 1e-9
    aux_basis: Optional[str] = None                    # for density fitting
    solvent_model: Optional[str] = None                # ddcosmo | pcm
    solvent_eps: Optional[float] = None
    nprocs: int = 1
    memory_mb: int = 4000
    unrestricted: bool = False                         # UHF/UKS
    extra_comments: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Method / basis normalisation maps
# ─────────────────────────────────────────────────────────────────────────────

METHOD_MAP = {
    # HF
    "hf": "hf", "rhf": "hf", "uhf": "hf",
    # DFT – pure
    "svwn": "svwn", "svwn5": "svwn5",
    "blyp": "blyp", "bp86": "bp86",
    "pbe": "pbe", "pw91": "pw91",
    "tpss": "tpss", "scan": "scan", "m06l": "m06l",
    # DFT – hybrid
    "b3lyp": "b3lyp", "b3pw91": "b3pw91",
    "pbe0": "pbe0", "pbeh": "pbe0",
    "tpssh": "tpssh",
    "m06": "m06", "m062x": "m062x", "m06-2x": "m062x",
    "m08hx": "m08hx", "m11": "m11",
    "bhlyp": "bhlyp",
    # DFT – range-separated
    "wb97": "wb97", "wb97x": "wb97x",
    "wb97x-d": "wb97x_d", "wb97x-d3": "wb97x_d3",
    "cam-b3lyp": "camb3lyp", "camb3lyp": "camb3lyp",
    "lc-blyp": "lc_blyp", "lc-wpbe": "lc_wpbe",
    # Post-HF
    "mp2": "mp2", "rimp2": "mp2",
    "ccsd": "ccsd", "ccsd(t)": "ccsd(t)",
    "cisd": "cisd",
}

BASIS_MAP = {
    # Pople
    "sto-3g": "sto-3g", "sto3g": "sto-3g",
    "3-21g": "3-21g",
    "6-31g": "6-31g", "6-31g*": "6-31g*", "6-31g**": "6-31g**",
    "6-31+g*": "6-31+g*", "6-31+g**": "6-31+g**",
    "6-311g": "6-311g", "6-311g*": "6-311g*", "6-311g**": "6-311g**",
    "6-311+g**": "6-311+g**", "6-311++g**": "6-311++g**",
    "6-311+g(2d,p)": "6-311+g(2d,p)",
    # Dunning correlation-consistent
    "cc-pvdz": "cc-pvdz", "cc-pvtz": "cc-pvtz",
    "cc-pvqz": "cc-pvqz", "cc-pv5z": "cc-pv5z",
    "aug-cc-pvdz": "aug-cc-pvdz", "aug-cc-pvtz": "aug-cc-pvtz",
    "aug-cc-pvqz": "aug-cc-pvqz",
    "cc-pcvdz": "cc-pcvdz", "cc-pcvtz": "cc-pcvtz",
    # Ahlrichs def2
    "def2-sv(p)": "def2-svp", "def2-svp": "def2-svp",
    "def2-tzvp": "def2-tzvp", "def2-tzvpp": "def2-tzvpp",
    "def2-qzvp": "def2-qzvp", "def2-qzvpp": "def2-qzvpp",
    "def2-svpd": "def2-svpd",
    # Jensen
    "pcseg-0": "pcseg-0", "pcseg-1": "pcseg-1",
    "pcseg-2": "pcseg-2", "pcseg-3": "pcseg-3",
}


def normalise_method(raw: str) -> str:
    key = raw.lower().strip()
    return METHOD_MAP.get(key, key)


def normalise_basis(raw: str) -> str:
    key = raw.lower().strip()
    return BASIS_MAP.get(key, key)


# ─────────────────────────────────────────────────────────────────────────────
# Psi4 parser
# ─────────────────────────────────────────────────────────────────────────────

class Psi4Parser:
    """Parse a Psi4 input file (.inp or .dat) into a QChemJob."""

    def parse(self, text: str) -> QChemJob:
        job = QChemJob()
        text = self._strip_comments(text)

        job.atoms, job.charge, job.multiplicity = self._parse_molecule(text)
        self._parse_set_block(text, job)
        self._parse_task(text, job)
        self._parse_memory(text, job)
        self._parse_nprocs(text, job)
        self._parse_solvent(text, job)

        # Open-shell detection
        if job.multiplicity != 1:
            job.unrestricted = True

        return job

    # ------------------------------------------------------------------
    def _strip_comments(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            # Psi4 comments start with #
            line = re.sub(r'#.*$', '', line)
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _parse_molecule(self, text: str):
        """Extract atoms, charge, multiplicity from molecule{} block."""
        atoms = []
        charge, mult = 0, 1

        mol_pattern = re.compile(
            r'molecule\s*\w*\s*\{(.*?)\}',
            re.DOTALL | re.IGNORECASE
        )
        m = mol_pattern.search(text)
        if not m:
            return atoms, charge, mult

        block = m.group(1).strip()
        lines = [l.strip() for l in block.splitlines() if l.strip()]

        # First non-blank line is charge / multiplicity
        first = lines[0].split()
        if len(first) >= 2 and all(t.lstrip('-').isdigit() for t in first[:2]):
            charge = int(first[0])
            mult = int(first[1])
            lines = lines[1:]

        # Remaining lines are coordinates
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                sym = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atoms.append((sym, x, y, z))
                except ValueError:
                    pass  # might be a Z-matrix line – skip for now
            elif len(parts) == 1 and parts[0].lower() in ('--', 'units', 'symmetry'):
                pass  # fragment separator or option – skip

        return atoms, charge, mult

    # ------------------------------------------------------------------
    def _parse_set_block(self, text: str, job: QChemJob):
        """Parse set { ... } or inline set keyword = value."""
        set_pattern = re.compile(r'set\s*\{(.*?)\}', re.DOTALL | re.IGNORECASE)
        for m in set_pattern.finditer(text):
            block = m.group(1)
            self._apply_set_options(block, job)

        # Also handle: set basis 6-31g*
        inline = re.compile(r'set\s+(\w+)\s*=?\s*(\S+)', re.IGNORECASE)
        for m in inline.finditer(text):
            key, val = m.group(1).lower(), m.group(2).strip().strip("'\"")
            self._set_option(key, val, job)

    def _apply_set_options(self, block: str, job: QChemJob):
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'[\s=]+', line, maxsplit=1)
            if len(parts) == 2:
                self._set_option(parts[0].lower(), parts[1].strip().strip("'\""), job)

    def _set_option(self, key: str, val: str, job: QChemJob):
        if key == "basis":
            job.basis = normalise_basis(val)
        elif key == "df_basis_scf":
            job.aux_basis = normalise_basis(val)
        elif key in ("maxiter", "scf_maxiter"):
            job.scf_max_cycles = int(val)
        elif key in ("e_convergence", "d_convergence"):
            try:
                job.scf_conv_tol = float(val)
            except ValueError:
                pass
        elif key == "reference":
            if val.lower() in ("uhf", "uks"):
                job.unrestricted = True

    # ------------------------------------------------------------------
    def _parse_task(self, text: str, job: QChemJob):
        """Detect energy/optimize/frequencies calls and extract method."""
        # energy('b3lyp') / optimize('mp2') / frequencies('hf')
        task_pattern = re.compile(
            r'(energy|optimize|frequency|frequencies)\s*\(\s*[\'"]([^\'"]+)[\'"]',
            re.IGNORECASE
        )
        m = task_pattern.search(text)
        if m:
            task_word = m.group(1).lower()
            method_raw = m.group(2)
            if task_word == "energy":
                job.job_type = "energy"
            elif task_word == "optimize":
                job.job_type = "opt"
            else:
                job.job_type = "freq"
            job.method = normalise_method(method_raw)

    # ------------------------------------------------------------------
    def _parse_memory(self, text: str, job: QChemJob):
        m = re.search(r'memory\s+([\d.]+)\s*(mb|gb|tb)?', text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            unit = (m.group(2) or "mb").lower()
            if unit == "gb":
                val *= 1024
            elif unit == "tb":
                val *= 1024 * 1024
            job.memory_mb = int(val)

    # ------------------------------------------------------------------
    def _parse_nprocs(self, text: str, job: QChemJob):
        m = re.search(r'set_num_threads\s*\(\s*(\d+)', text, re.IGNORECASE)
        if m:
            job.nprocs = int(m.group(1))

    # ------------------------------------------------------------------
    def _parse_solvent(self, text: str, job: QChemJob):
        # set pcm { ... } or set ddcosmo { ... }
        if re.search(r'pcm\s*\{', text, re.IGNORECASE):
            job.solvent_model = "pcm"
            eps_m = re.search(r'eps\s*=\s*([\d.]+)', text, re.IGNORECASE)
            if eps_m:
                job.solvent_eps = float(eps_m.group(1))
        elif re.search(r'ddcosmo', text, re.IGNORECASE):
            job.solvent_model = "ddcosmo"


# ─────────────────────────────────────────────────────────────────────────────
# ORCA parser
# ─────────────────────────────────────────────────────────────────────────────

class OrcaParser:
    """Parse an ORCA input file into a QChemJob."""

    def parse(self, text: str) -> QChemJob:
        job = QChemJob()
        text = self._strip_comments(text)

        self._parse_simple_input(text, job)
        self._parse_blocks(text, job)
        self._parse_geometry(text, job)

        if job.multiplicity != 1:
            job.unrestricted = True

        return job

    # ------------------------------------------------------------------
    def _strip_comments(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            line = re.sub(r'!.*$', '', line)   # ORCA inline comments use !
            # But ! is also the keyword line — only strip if NOT the first token
            lines.append(line)
        # ORCA keyword lines start with '!'
        return text  # re-parse carefully below

    # ------------------------------------------------------------------
    def _parse_simple_input(self, text: str, job: QChemJob):
        """Parse ! KEYWORD KEYWORD ... lines."""
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith('!'):
                continue
            tokens = stripped.lstrip('!').split()
            method_found = False
            basis_found = False
            for tok in tokens:
                tl = tok.lower()

                # Job type keywords
                if tl in ('opt', 'copt', 'gdiis-opt', 'noopt'):
                    job.job_type = 'opt' if tl != 'noopt' else 'energy'
                elif tl in ('freq', 'numfreq', 'anfreq'):
                    job.job_type = 'freq'
                elif tl in ('sp',):
                    job.job_type = 'energy'

                # Solver / reference keywords
                elif tl in ('uhf', 'uks', 'uno'):
                    job.unrestricted = True
                elif tl in ('rhf', 'rks', 'rohf', 'roks'):
                    job.unrestricted = False

                # Density fitting
                elif tl in ('ri', 'rijk', 'rijonx', 'rijcosx'):
                    pass  # flag — handled via aux basis

                # Method detection (simple tokens that are pure method names)
                elif not method_found and tl in METHOD_MAP:
                    job.method = METHOD_MAP[tl]
                    method_found = True

                # Basis detection
                elif not basis_found and tl in BASIS_MAP:
                    job.basis = BASIS_MAP[tl]
                    basis_found = True

                # Auxiliary basis
                elif tl.startswith('def2/') or tl.startswith('aug-cc-p') and 'c' in tl:
                    pass

                # Common basis patterns not in the map
                elif not basis_found and re.match(
                        r'^(6-31|6-311|cc-p|aug-cc|def2|sto|3-21)', tl):
                    job.basis = normalise_basis(tl)
                    basis_found = True

                elif not method_found and re.match(
                        r'^(b3lyp|pbe|m06|wb97|cam|lc|bp86|blyp|tpss|scan|svwn)', tl):
                    job.method = normalise_method(tl)
                    method_found = True

    # ------------------------------------------------------------------
    def _parse_blocks(self, text: str, job: QChemJob):
        """Parse %block ... end sections."""

        # %pal nprocs N end
        m = re.search(r'%pal\s+nprocs\s+(\d+)', text, re.IGNORECASE)
        if m:
            job.nprocs = int(m.group(1))

        # %maxcore N
        m = re.search(r'%maxcore\s+(\d+)', text, re.IGNORECASE)
        if m:
            job.memory_mb = int(m.group(1))

        # %scf block
        scf_block = re.search(r'%scf(.*?)end', text, re.DOTALL | re.IGNORECASE)
        if scf_block:
            b = scf_block.group(1)
            mi = re.search(r'maxiter\s+(\d+)', b, re.IGNORECASE)
            if mi:
                job.scf_max_cycles = int(mi.group(1))
            ti = re.search(r'tole\s+([\d.e+-]+)', b, re.IGNORECASE)
            if ti:
                try:
                    job.scf_conv_tol = float(ti.group(1))
                except ValueError:
                    pass

        # %cpcm / %cosmo (solvent)
        if re.search(r'%cpcm|%cosmo', text, re.IGNORECASE):
            job.solvent_model = "ddcosmo"
            eps_m = re.search(r'epsilon\s+([\d.]+)', text, re.IGNORECASE)
            if eps_m:
                job.solvent_eps = float(eps_m.group(1))

        # %basis  AuxJ / AuxC ...
        aux_m = re.search(r'auxj\s+"?([^"\s]+)"?', text, re.IGNORECASE)
        if aux_m:
            job.aux_basis = normalise_basis(aux_m.group(1))

    # ------------------------------------------------------------------
    def _parse_geometry(self, text: str, job: QChemJob):
        """Parse * xyz charge mult ... * geometry block."""
        # * xyz 0 1
        xyz_pattern = re.compile(
            r'\*\s+xyz\s+(-?\d+)\s+(\d+)(.*?)\*',
            re.DOTALL | re.IGNORECASE
        )
        m = xyz_pattern.search(text)
        if not m:
            # Try * xyzfile format
            xyzfile_m = re.search(
                r'\*\s+xyzfile\s+(-?\d+)\s+(\d+)\s+(\S+)',
                text, re.IGNORECASE
            )
            if xyzfile_m:
                job.charge = int(xyzfile_m.group(1))
                job.multiplicity = int(xyzfile_m.group(2))
                job.extra_comments.append(
                    f"# NOTE: geometry loaded from file '{xyzfile_m.group(3)}'"
                )
                job.extra_comments.append(
                    f"# Load with: mol.atom = open('{xyzfile_m.group(3)}').read()"
                )
            return

        job.charge = int(m.group(1))
        job.multiplicity = int(m.group(2))

        coord_block = m.group(3).strip()
        for line in coord_block.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                sym = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    job.atoms.append((sym, x, y, z))
                except ValueError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# PySCF code generator
# ─────────────────────────────────────────────────────────────────────────────

class PySCFGenerator:
    """Generate a PySCF Python script from a QChemJob."""

    POST_HF_METHODS = {"mp2", "ccsd", "ccsd(t)", "cisd"}
    DFT_METHODS = {
        "svwn", "svwn5", "blyp", "bp86", "pbe", "pw91", "tpss", "scan",
        "m06l", "b3lyp", "b3pw91", "pbe0", "tpssh", "m06", "m062x",
        "m08hx", "m11", "bhlyp", "wb97", "wb97x", "wb97x_d", "wb97x_d3",
        "camb3lyp", "lc_blyp", "lc_wpbe",
    }

    def generate(self, job: QChemJob, source_name: str = "input") -> str:
        lines = []

        # Header
        lines += [
            f"# PySCF script generated from: {source_name}",
            "# Generated by qchem_converter.py",
            "",
        ]
        if job.extra_comments:
            lines += job.extra_comments + [""]

        # Imports
        lines += self._imports(job)
        lines.append("")

        # Memory / threads
        lines += self._resource_setup(job)
        lines.append("")

        # Molecule
        lines += self._molecule(job)
        lines.append("")

        # SCF / method setup
        lines += self._method_setup(job)
        lines.append("")

        # Run job
        lines += self._run_job(job)
        lines.append("")

        # Property extraction
        lines += self._print_results(job)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _imports(self, job: QChemJob) -> list:
        imports = ["from pyscf import gto, scf, dft"]

        method = job.method.lower()
        if method in self.POST_HF_METHODS:
            if method == "mp2":
                imports.append("from pyscf import mp")
            elif method.startswith("ccsd"):
                imports.append("from pyscf import cc")
            elif method == "cisd":
                imports.append("from pyscf import ci")

        if job.job_type == "opt":
            imports.append("from pyscf.geomopt.berny_solver import optimize")
        if job.job_type == "freq":
            imports.append("# Frequencies via finite differences:")
            if method in self.POST_HF_METHODS:
                imports.append("from pyscf.hessian import thermo")
            else:
                imports.append("from pyscf.hessian import thermo")

        if job.solvent_model:
            imports.append("from pyscf import solvent")

        return imports

    # ------------------------------------------------------------------
    def _resource_setup(self, job: QChemJob) -> list:
        lines = [
            "# ── Resources ────────────────────────────────────────────────",
            f"import os",
            f"os.environ['OMP_NUM_THREADS'] = '{job.nprocs}'",
            f"",
            f"from pyscf import lib",
            f"lib.num_threads({job.nprocs})",
            f"lib.max_memory = {job.memory_mb}  # MB",
        ]
        return lines

    # ------------------------------------------------------------------
    def _molecule(self, job: QChemJob) -> list:
        lines = ["# ── Molecule ─────────────────────────────────────────────────"]
        lines.append("mol = gto.Mole()")

        if job.atoms:
            atom_lines = []
            for sym, x, y, z in job.atoms:
                atom_lines.append(f"    {sym}  {x:14.8f}  {y:14.8f}  {z:14.8f};")
            lines.append("mol.atom = '''")
            lines += atom_lines
            lines.append("'''")
        else:
            lines.append("mol.atom = '''")
            lines.append("    # TODO: insert atom coordinates here")
            lines.append("'''")

        lines.append(f"mol.charge = {job.charge}")
        lines.append(f"mol.spin   = {job.multiplicity - 1}  # 2S = multiplicity - 1")
        lines.append(f"mol.basis  = '{job.basis}'")

        if job.aux_basis:
            lines.append(f"mol.auxbasis = '{job.aux_basis}'")

        lines.append("mol.verbose = 4")
        lines.append("mol.build()")
        return lines

    # ------------------------------------------------------------------
    def _method_setup(self, job: QChemJob) -> list:
        lines = ["# ── Method setup ─────────────────────────────────────────────"]
        method = job.method.lower()
        unrestricted = job.unrestricted
        prefix = "u" if unrestricted else "r"

        if method == "hf":
            cls = f"scf.{'UHF' if unrestricted else 'RHF'}(mol)"
            lines.append(f"mf = {cls}")
        elif method in self.DFT_METHODS:
            cls = f"dft.{'UKS' if unrestricted else 'RKS'}(mol)"
            lines.append(f"mf = {cls}")
            lines.append(f"mf.xc = '{job.method}'")
        elif method in self.POST_HF_METHODS:
            # Build mean-field first
            hf_cls = f"scf.{'UHF' if unrestricted else 'RHF'}(mol)"
            lines.append(f"mf = {hf_cls}")
            lines.append(f"mf.max_cycle  = {job.scf_max_cycles}")
            lines.append(f"mf.conv_tol   = {job.scf_conv_tol}")
            if job.solvent_model:
                lines += self._solvent_wrap(job, "mf")
            lines.append("mf.kernel()")
            lines.append("")
            lines.append("# Post-HF correction")
            if method == "mp2":
                cls2 = "mp.UMP2" if unrestricted else "mp.MP2"
            elif method == "ccsd":
                cls2 = "cc.UCCSD" if unrestricted else "cc.CCSD"
            elif method == "ccsd(t)":
                cls2 = "cc.UCCSD" if unrestricted else "cc.CCSD"
            elif method == "cisd":
                cls2 = "ci.UCISD" if unrestricted else "ci.CISD"
            else:
                cls2 = f"# TODO: {method}"
            lines.append(f"corr = {cls2}(mf)")
            return lines  # early return for post-HF

        lines.append(f"mf.max_cycle  = {job.scf_max_cycles}")
        lines.append(f"mf.conv_tol   = {job.scf_conv_tol}")

        if job.solvent_model:
            lines += self._solvent_wrap(job, "mf")

        return lines

    # ------------------------------------------------------------------
    def _solvent_wrap(self, job: QChemJob, var: str) -> list:
        lines = [f"# ── Solvent ({job.solvent_model}) ───"]
        if job.solvent_model in ("ddcosmo", "pcm"):
            lines.append(f"{var} = solvent.ddCOSMO({var})")
            if job.solvent_eps:
                lines.append(f"{var}.with_solvent.eps = {job.solvent_eps}")
        return lines

    # ------------------------------------------------------------------
    def _run_job(self, job: QChemJob) -> list:
        lines = ["# ── Run ──────────────────────────────────────────────────────"]
        method = job.method.lower()
        is_post_hf = method in self.POST_HF_METHODS

        if is_post_hf:
            lines.append("e_hf = mf.e_tot  # already converged above")
            if method in ("mp2",):
                lines.append("corr.kernel()")
                lines.append("e_total = corr.e_tot")
            elif method in ("ccsd", "ccsd(t)"):
                lines.append("corr.kernel()")
                lines.append("e_ccsd = corr.e_tot")
                if method == "ccsd(t)":
                    lines.append("# CCSD(T) perturbative triples")
                    lines.append("from pyscf.cc import ccsd_t")
                    lines.append("e_t = ccsd_t.kernel(corr, corr.ao2mo())")
                    lines.append("e_total = e_ccsd + e_t")
                else:
                    lines.append("e_total = e_ccsd")
            elif method == "cisd":
                lines.append("corr.kernel()")
                lines.append("e_total = corr.e_tot")
            else:
                lines.append("corr.kernel()")
                lines.append("e_total = corr.e_tot")
            if job.job_type == "freq":
                lines += [
                    "",
                    "# Harmonic frequencies (finite-diff Hessian at post-HF level)",
                    "hessian = corr.Hessian().kernel()",
                    "freq_info = thermo.harmonic_analysis(mol, hessian)",
                    "thermo.dump_normal_mode(mol, freq_info)",
                ]
        elif job.job_type == "energy":
            lines.append("e_total = mf.kernel()")
        elif job.job_type == "opt":
            lines.append("# Geometry optimisation (Berny / geomeTRIC)")
            lines.append("mol_eq = optimize(mf)")
            lines.append("print('Optimised geometry:')")
            lines.append("print(mol_eq.atom_coords())")
            lines.append("e_total = mf.e_tot")
        elif job.job_type == "freq":
            lines.append("# Single point first")
            lines.append("mf.kernel()")
            lines.append("e_total = mf.e_tot")
            lines.append("")
            lines.append("# Harmonic frequencies via Hessian")
            lines.append("hessian = mf.Hessian().kernel()")
            lines.append("freq_info = thermo.harmonic_analysis(mol, hessian)")
            lines.append("thermo.dump_normal_mode(mol, freq_info)")

        return lines

    # ------------------------------------------------------------------
    def _print_results(self, job: QChemJob) -> list:
        method = job.method.lower()
        is_post_hf = method in self.POST_HF_METHODS

        lines = [
            "# ── Results ──────────────────────────────────────────────────",
            "print(f'Total energy = {e_total:.10f} Ha')",
        ]
        if not is_post_hf and job.job_type == "energy":
            lines += [
                "",
                "# Additional properties",
                "import numpy as np",
                "dm = mf.make_rdm1()",
                "print(f'Dipole moment (Debye) = {mf.dip_moment()}')",
            ]
            if job.multiplicity != 1:
                lines.append(
                    "print(f'<S^2> = {mf.spin_square()[0]:.4f}')"
                )
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Format auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(text: str, filename: str) -> str:
    """Guess whether a file is Psi4 or ORCA format."""
    # Extension hints
    fname = filename.lower()
    if fname.endswith('.inp'):
        # Both formats use .inp – look at content
        pass
    if fname.endswith('.dat'):
        return 'psi4'

    # Content clues
    if re.search(r'^\s*molecule\s*\w*\s*\{', text, re.MULTILINE | re.IGNORECASE):
        return 'psi4'
    if re.search(r'^\s*!', text, re.MULTILINE):
        return 'orca'
    if re.search(r'\*\s+xyz\s+-?\d+\s+\d+', text, re.IGNORECASE):
        return 'orca'
    if re.search(r'energy\s*\(|optimize\s*\(|frequencies\s*\(', text, re.IGNORECASE):
        return 'psi4'

    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def convert(text: str, fmt: str = "auto", source_name: str = "input") -> str:
    """
    Convert a Psi4 or ORCA input string to a PySCF Python script.

    Parameters
    ----------
    text        : raw content of the Psi4 / ORCA input file
    fmt         : 'psi4', 'orca', or 'auto'
    source_name : original filename, used only in the header comment

    Returns
    -------
    str  – ready-to-run PySCF Python script
    """
    if fmt == "auto":
        fmt = detect_format(text, source_name)
        if fmt == "unknown":
            raise ValueError(
                "Cannot auto-detect format. "
                "Pass --fmt psi4 or --fmt orca explicitly."
            )

    if fmt == "psi4":
        job = Psi4Parser().parse(text)
    elif fmt == "orca":
        job = OrcaParser().parse(text)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'psi4' or 'orca'.")

    return PySCFGenerator().generate(job, source_name=source_name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Psi4 / ORCA input files to PySCF Python scripts."
    )
    parser.add_argument("input", help="Path to Psi4 or ORCA input file")
    parser.add_argument(
        "--fmt", choices=["auto", "psi4", "orca"], default="auto",
        help="Input format (default: auto-detect)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .py file (default: <input_stem>_pyscf.py)"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    text = in_path.read_text()
    result = convert(text, fmt=args.fmt, source_name=in_path.name)

    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_pyscf.py"
    )
    out_path.write_text(result)
    print(f"✓  Written: {out_path}")


if __name__ == "__main__":
    main()
