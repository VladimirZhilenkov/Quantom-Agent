"""Tool function registry → Claude JSON schemas."""


def register_tool(func):
    """Decorator to register a function as an agent tool."""
    raise NotImplementedError


def get_tool_schemas() -> list[dict]:
    """Return Claude-compatible JSON schemas for all registered tools."""
    raise NotImplementedError
