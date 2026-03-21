# qchem-agent

> AI agent for automating quantum chemistry calculations.  
> Natural language in → PySCF / ORCA / Psi4 calculations → structured results + reports.

---

## What we are building

An LLM-powered agent that lets a chemist describe a calculation in plain language and
receives back energies, optimised geometries, spectra, and a human-readable report —
without writing a single input file or parsing raw output manually.

**Example interaction**

```
User:  Calculate the dipole moment of water at B3LYP/def2-TZVP and optimise the geometry.

Agent: [selects method] → [generates PySCF script] → [runs calculation] →
       [parses output] → [renders 3D geometry] → [writes report]

Result: E = -76.4231 Ha | Dipole = 1.854 D | Geometry converged in 12 steps
        [inline 3D viewer] [download report.md]
```

---

## Team

| Name     | Role                                                         |
|----------|--------------------------------------------------------------|
| Vladimir | QM domain lead — chemical calculation tools, method selection |
| Misha    | Output parsing, molecule visualisation, report generation    |
| Vlad     | LLM API, agent framework, tool access layer, frontend        |
| Ihsaan   | Databases, async job dispatch, vector store, training mgmt   |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User interface                      │
│              CLI (Typer)  ·  REST API  ·  Web UI        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  Agent core  (Vlad)                     │
│   LangGraph  ·  Anthropic SDK  ·  Tool registry         │
│   Think → Act → Observe loop  ·  Error recovery         │
└────┬──────────────────┬──────────────────┬──────────────┘
     │                  │                  │
┌────▼─────┐    ┌───────▼──────┐   ┌──────▼───────┐
│ QM tools │    │    Parser /  │   │   Database / │
│(Vladimir)│    │  Visualiser  │   │    Async     │
│          │    │   (Misha)    │   │  (Ihsaan)    │
│ PySCF    │    │ cclib        │   │ SQLAlchemy   │
│ ORCA     │    │ py3Dmol      │   │ ChromaDB     │
│ Psi4     │    │ RDKit        │   │ asyncio      │
│ ASE      │    │ Matplotlib   │   │ MLflow       │
│ RDKit    │    │ Jinja2       │   │              │
└────┬─────┘    └───────┬──────┘   └──────┬───────┘
     │                  │                  │
     └──────────────────▼──────────────────┘
                   Shared schemas
                   Pydantic models
                   (schemas.py)
```

---

## Repository structure

```
qchem-agent/
│
├── schemas.py                  # Shared Pydantic models — QChemJob, QChemResult
│
├── tools/                      # Vladimir + Misha
│   ├── run_pyscf.py            # Execute PySCF scripts, return QChemResult
│   ├── run_orca.py             # Execute ORCA via subprocess, return QChemResult
│   ├── run_psi4.py             # Execute Psi4 via Python API, return QChemResult
│   ├── parse_output.py         # cclib-based output parser → QChemResult
│   ├── parse_properties.py     # NBO, NMR, UV-vis, IR parsing (Phase 4)
│   ├── visualise.py            # py3Dmol / RDKit → PNG / HTML / SVG
│   ├── report.py               # Jinja2 report generator → markdown
│   └── method_selector.py      # RAG-powered method + basis recommender
│
├── converter/                  # Vladimir — DONE ✅
│   ├── qchem_converter.py      # Psi4 / ORCA → PySCF converter (83 tests)
│   └── test_converter.py
│
├── agent/                      # Vlad
│   ├── llm.py                  # Anthropic SDK — tool_use call/response cycle
│   ├── registry.py             # Tool function registry → Claude JSON schemas
│   ├── graph.py                # LangGraph Think → Act → Observe state graph
│   ├── planner.py              # Multi-step job decomposition (Phase 3)
│   └── error_recovery.py       # Failure classification + retry heuristics
│
├── db/                         # Ihsaan
│   ├── models.py               # SQLAlchemy ORM — Job, Result tables
│   ├── crud.py                 # create_job(), update_status(), get_job()
│   └── migrations/             # Alembic migration scripts
│
├── worker/                     # Ihsaan
│   ├── tasks.py                # Async job dispatch — submit_job(), check_status()
│   └── slurm.py                # HPC Slurm integration (Phase 4)
│
├── rag/                        # Ihsaan
│   ├── store.py                # ChromaDB setup + literature ingestion
│   └── search.py               # search_literature(query) tool
│
├── training/                   # Ihsaan
│   └── tracker.py              # MLflow experiment tracking
│
├── api/                        # Vlad
│   └── main.py                 # FastAPI — /jobs, /status, /results endpoints
│
├── cli/                        # Vlad
│   └── main.py                 # Typer CLI — qchem-agent run / status / results
│
├── frontend/                   # Vlad + Misha (Phase 4)
│   └── app.py                  # Streamlit web UI
│
├── eval/                       # Vladimir + Misha (Phase 3)
│   └── benchmark.py            # W4-17 / GMTKN55 / NIST CCCBDB harness
│
├── templates/                  # Misha
│   └── report.md.j2            # Jinja2 report template
│
├── tests/
│   ├── test_tools.py
│   ├── test_agent.py
│   ├── test_parser.py
│   └── test_db.py
│
├── docker/
│   ├── Dockerfile.pyscf        # PySCF + GPU4PySCF
│   ├── Dockerfile.orca         # ORCA 6 (requires license)
│   └── Dockerfile.agent        # Agent + API + CLI
│
├── docker-compose.yml          # Full local stack
├── pyproject.toml
├── .env.example
└── AGENTS.md                   # ← this file
```

---

## Shared data models  (`schemas.py`)

These are the central contracts. **All tools, parsers, and DB layers use these.**
Defined by Vladimir + Misha together in Phase 1.

```python
from pydantic import BaseModel
from typing import Literal

class QChemJob(BaseModel):
    id: str
    method: str                                     # e.g. "b3lyp", "ccsd(t)"
    basis: str                                      # e.g. "def2-tzvp"
    charge: int
    multiplicity: int
    atoms: list[tuple[str, float, float, float]]    # (symbol, x, y, z) in Å
    job_type: Literal["energy", "opt", "freq"]
    engine: Literal["pyscf", "orca", "psi4"] = "pyscf"
    solvent: str | None = None
    nprocs: int = 1
    memory_mb: int = 4000

class QChemResult(BaseModel):
    job_id: str
    energy: float | None                            # total energy in Hartree
    converged: bool
    geometry: list[tuple[str, float, float, float]] | None
    frequencies: list[float] | None                 # in cm⁻¹
    dipole: tuple[float, float, float] | None       # Debye
    wall_time: float                                # seconds
    engine: str
    method: str
    basis: str
    raw_output: str
    output_path: str | None = None
```

---

## Tool interface contract

Every tool callable by the agent follows this pattern:

```python
# Input: Pydantic model (auto-generates Claude JSON schema)
# Output: dict or Pydantic model
# Errors: raise typed exceptions — never return error strings

def run_pyscf(job: QChemJob) -> QChemResult: ...
def run_orca(job: QChemJob) -> QChemResult: ...
def parse_output(filepath: str) -> QChemResult: ...
def visualise_molecule(geometry: list, fmt: str = "png") -> bytes: ...
def generate_report(results: list[QChemResult], title: str) -> str: ...
def search_literature(query: str, n: int = 5) -> list[str]: ...
def submit_job(job: QChemJob) -> str: ...             # returns job_id
def check_status(job_id: str) -> str: ...             # pending|running|done|failed
def get_result(job_id: str) -> QChemResult: ...
```

---

## Libraries

### Vladimir — QM tools
| Library  | Version | Purpose                              |
|----------|---------|--------------------------------------|
| pyscf    | ≥2.9    | Primary QM engine                    |
| ase      | ≥3.23   | Geometry IO, structure manipulation  |
| rdkit    | ≥2024.3 | SMILES → 3D, mol manipulation        |
| pydantic | ≥2.0    | QChemJob / QChemResult schemas       |
| xtb      | ≥6.7    | Fast semi-empirical pre-screening    |

ORCA and Psi4 are external binaries — called via subprocess / Python API.

### Misha — parsing, visualisation, reports
| Library    | Version | Purpose                              |
|------------|---------|--------------------------------------|
| cclib      | ≥1.8    | Parse ORCA / Psi4 / Gaussian outputs |
| py3dmol    | ≥2.4    | Interactive 3D molecule viewer       |
| rdkit      | ≥2024.3 | 2D structure depiction, SVG          |
| matplotlib | ≥3.9    | Spectra, energy diagrams, PES        |
| jinja2     | ≥3.1    | Templated report generation          |
| pydantic   | ≥2.0    | Consume QChemResult schema           |

### Vlad — agent, API, frontend
| Library       | Version | Purpose                              |
|---------------|---------|--------------------------------------|
| anthropic     | ≥0.30   | Claude API — tool_use, streaming     |
| langgraph     | ≥0.2    | Stateful agent graph, retry loops    |
| langchain     | ≥0.3    | Tool wrappers, prompt templates      |
| fastapi       | ≥0.115  | REST API                             |
| uvicorn       | ≥0.32   | ASGI server                          |
| typer         | ≥0.12   | CLI                                  |
| langsmith     | ≥0.1    | LLM call tracing + observability     |
| pydantic      | ≥2.0    | Tool input/output schemas            |
| streamlit     | ≥1.40   | Web frontend (Phase 4)               |

### Ihsaan — database, async, training
| Library      | Version | Purpose                              |
|--------------|---------|--------------------------------------|
| sqlalchemy   | ≥2.0    | ORM for job store                    |
| alembic      | ≥1.14   | Database schema migrations           |
| chromadb     | ≥0.6    | Vector store for literature RAG      |
| mlflow       | ≥2.18   | Training experiment tracking         |
| pydantic     | ≥2.0    | Validate job payloads before DB      |

### Shared by everyone
| Library      | Purpose                       |
|--------------|-------------------------------|
| pydantic     | Central data contract         |
| pytest       | Unit and integration tests    |
| python-dotenv| API key and config management |
| loguru       | Structured logging            |

---

## Docker setup

The full stack runs in Docker. No external services required.

```yaml
# docker-compose.yml (simplified)
services:

  agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.agent
    volumes:
      - ./jobs:/app/jobs          # job input/output files
      - ./data:/app/data          # SQLite DB + ChromaDB
    env_file: .env
    ports:
      - "8000:8000"               # FastAPI
      - "8501:8501"               # Streamlit (Phase 4)
    depends_on:
      - pyscf

  pyscf:
    build:
      context: .
      dockerfile: docker/Dockerfile.pyscf
    volumes:
      - ./jobs:/jobs
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # optional — GPU4PySCF

  orca:
    build:
      context: .
      dockerfile: docker/Dockerfile.orca
    volumes:
      - ./jobs:/jobs
      - ./orca_license:/orca_license  # ORCA requires a license file
```

```
# .env.example
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=qchem-agent
DATABASE_URL=sqlite:///./data/jobs.db
CHROMA_PATH=./data/chroma
JOBS_DIR=./jobs
ORCA_BINARY=/opt/orca/orca
PSI4_BINARY=/opt/psi4/bin/psi4
LOG_LEVEL=INFO
```

**Local development (without Docker)**

```bash
git clone https://github.com/your-org/qchem-agent
cd qchem-agent
pip install -e ".[dev]"
cp .env.example .env    # fill in your keys
python -m pytest        # run tests
qchem-agent run "calculate energy of water at HF/STO-3G"
```

---

## Build phases

### Phase 1 — Foundation  *(build this first)*
| Task | Owner | Status |
|------|-------|--------|
| `qchem_converter.py` — Psi4/ORCA → PySCF | Vladimir | ✅ done |
| `schemas.py` — QChemJob + QChemResult | Vladimir + Misha | 🔲 |
| `run_pyscf()` tool wrapper | Vladimir | 🔲 |
| `run_orca()` tool wrapper | Vladimir | 🔲 |
| `parse_output()` with cclib | Misha | 🔲 |
| `visualise_molecule()` | Misha | 🔲 |
| Job store — SQLAlchemy + SQLite | Ihsaan | 🔲 |
| Claude API + tool_use integration | Vlad | 🔲 |

### Phase 2 — Agent core
- Tool registry (Vlad)
- LangGraph Think → Act → Observe loop (Vlad)
- Error recovery node (Vlad + Vladimir)
- Async job dispatch with asyncio (Ihsaan)
- FastAPI REST endpoints (Vlad)
- CLI with Typer (Vlad)
- Report generation pipeline (Misha)

### Phase 3 — Intelligence
- ChromaDB + literature RAG (Ihsaan)
- RAG-powered method recommender (Ihsaan + Vladimir)
- Benchmark evaluation — W4-17, GMTKN55 (Vladimir + Misha)
- Multi-step planner (Vlad + Vladimir)
- LangSmith tracing (Vlad + Ihsaan)
- MLflow training tracker (Ihsaan)

### Phase 4 — Production
- Docker + Singularity environments (Ihsaan)
- HPC Slurm integration (Ihsaan + Vladimir)
- Jupyter notebook integration (Vlad + Misha)
- Streamlit web frontend (Vlad + Misha)
- Extended parser — NBO, NMR, UV-vis (Misha + Vladimir)

---

## Key design decisions

**Why PySCF as the primary engine?**  
PySCF is pure Python — it can be called directly as a library, no subprocess needed,
and it integrates naturally as a tool callable by the agent. ORCA and Psi4 are supported
via subprocess wrappers for cases where ORCA's open-shell DFT or Psi4's CBS
extrapolation is specifically needed.

**Async job dispatch without a task queue**  
Python's `asyncio` + `concurrent.futures` handles concurrent QM job dispatch without
any external broker infrastructure. The SQLite job store tracks job status.
Long-running ORCA or Psi4 jobs run as async subprocesses; the agent polls
`check_status(job_id)` without blocking the reasoning loop.

**Local filesystem for job files**  
Large checkpoint files (wfn, cube, molden) are stored on the local filesystem under
`jobs/{job_id}/`. A structured path convention is sufficient for a 4-person project
and works naturally inside Docker volumes.

**Why LangGraph over raw Anthropic SDK?**  
QM workflows loop — SCF convergence failures require retry with modified parameters,
geometry optimisations run over many steps, complex tasks need multi-step planning.
LangGraph models this as a directed graph with explicit retry edges, which fits the
domain perfectly.

**Double-hybrid DFT is handled explicitly.**  
The converter already implements a registry of 20 double-hybrid functionals
(B2PLYP, B2GP-PLYP, PWPB95, DSD-BLYP, DSD-PBEP86, PBE0-DH, PBE-QIDH, …)
with published mixing coefficients. The KS step uses an explicit xc string with
correct (1−ax)·DFA_X + ax·HF mixing, followed by scaled PT2 correlation on KS
orbitals. Spin-component scaling (SOS/SS) is handled correctly for PWPB95 and
DSD-* functionals.

---

## Related work

| System | Paper | Notes |
|--------|-------|-------|
| El Agente Q | Matter 8, 102263 (2025) | Multi-agent, ORCA focus, 87% task success |
| AMLP | JCTC (2025) | ORCA + Psi4, ML potential pipeline |
| ChemGraph | arXiv 2506.06363 (2025) | ASE-based, NWChem + ORCA |
| ASH framework | ash.readthedocs.io | Python automation layer, ORCA+PySCF+Psi4 |

**Our differentiator:** deep PySCF-native integration (underserved in existing agents),
double-hybrid DFT support, and a clean Python-only stack requiring no external services
for local use.
