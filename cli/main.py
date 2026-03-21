"""Typer CLI — qchem-agent run / status / results."""

import typer

app = typer.Typer(name="qchem-agent")


@app.command()
def run(prompt: str = typer.Argument(..., help="Describe the calculation in plain language")):
    """Run a quantum chemistry calculation from a natural language prompt."""
    raise NotImplementedError


@app.command()
def status(job_id: str = typer.Argument(..., help="Job ID to check")):
    """Check the status of a submitted job."""
    raise NotImplementedError


@app.command()
def result(job_id: str = typer.Argument(..., help="Job ID to retrieve")):
    """Retrieve the result of a completed job."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
