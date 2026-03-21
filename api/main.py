"""FastAPI — /jobs, /status, /results endpoints."""

from fastapi import FastAPI

app = FastAPI(title="qchem-agent", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /jobs — submit a new calculation
# TODO: GET  /jobs/{job_id}/status — check job status
# TODO: GET  /jobs/{job_id}/result — get calculation result
