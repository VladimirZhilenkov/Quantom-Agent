"""ChromaDB setup + literature ingestion."""


def init_store(path: str) -> None:
    """Initialize the ChromaDB vector store."""
    raise NotImplementedError


def ingest_documents(docs: list[str]) -> int:
    """Ingest literature documents into the vector store. Return count."""
    raise NotImplementedError
