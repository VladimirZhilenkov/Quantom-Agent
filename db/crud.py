"""CRUD operations for the job store."""


def create_job(job_data: dict) -> str:
    """Create a new job record, return job_id."""
    raise NotImplementedError


def update_status(job_id: str, status: str) -> None:
    """Update job status."""
    raise NotImplementedError


def get_job(job_id: str) -> dict:
    """Retrieve a job record by ID."""
    raise NotImplementedError
