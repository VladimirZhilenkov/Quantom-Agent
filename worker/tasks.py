"""Async job dispatch — submit_job(), check_status()."""

from schemas import QChemJob


def submit_job(job: QChemJob) -> str:
    """Submit a job for async execution, return job_id."""
    raise NotImplementedError


def check_status(job_id: str) -> str:
    """Check job status: pending | running | done | failed."""
    raise NotImplementedError
