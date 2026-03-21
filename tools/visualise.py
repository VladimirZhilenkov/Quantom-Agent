"""Molecule visualisation — py3Dmol / RDKit → PNG / HTML / SVG."""


def visualise_molecule(
    geometry: list[tuple[str, float, float, float]],
    fmt: str = "png",
) -> bytes:
    """Render a molecular geometry as an image or interactive viewer."""
    raise NotImplementedError
