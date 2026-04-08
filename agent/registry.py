from typing import Callable, Any
from pydantic import BaseModel, TypeAdapter, ValidationError
import inspect
from loguru import logger
from converter import qchem_converter

tools = {}

def register_tool(func: Callable):
    tools[func.__name__] = func
    logger.info(f"Tool {func.__name__} ready for work.")
    return func

def annotation_to_json_schema(annotation: Any) -> dict:
    try:
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            return annotation.model_json_schema()
        return TypeAdapter(annotation).json_schema()
    except Exception:
        return {"type": "string"}

def get_tool_schemas() -> list[dict]:
    schemas = []
    for name, func in tools.items():
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            properties[param_name] = annotation_to_json_schema(param.annotation)

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schemas.append({
            "name": name,
            "description": func.__doc__ or "Execute a quantum chemical calculation",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    return schemas

async def execute_tool(name: str, args: dict) -> Any:
    func = tools.get(name)
    if not func:
        return f"Error: Tool {name} not found."

    if not isinstance(args, dict):
        return f"Error: args must be a dictionary."
    
    try:
        if inspect.iscoroutinefunction(func):
            return await func(**args)
        return func(**args)
    except Exception as e:
        logger.exception(f"Tool '{name}' execution failed")
        return f"Error during execution: {str(e)}"


@register_tool
async def echo(message: str) -> str:
    return f"Echo response: {message}"

@register_tool 
def standardize_chem_input(raw_format: str, input_format: str = "auto") -> str:
    try:
        pyscf_script = qchem_converter.convert(
            text=raw_format,
            fmt=input_format,
            source_name="agent_standardizer"
        )

        if not pyscf_script:
            return "Error: Converter didn't create a script. Check data validity"
        return pyscf_script
    
    except ValueError as ve:
        return f"Recognize error: {str(ve)}. Specify format (orca/psi4)."
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return f"Conversion technical error: {str(e)}"