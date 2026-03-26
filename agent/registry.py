from typing import Callable, Any
from pydantic import BaseModel, TypeAdapter, ValidationError
import inspect
from loguru import logger

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
        return {}
    
    except Exception:
        logger.exception(f"Failed to build input schema for annotation: {annotation}")
        return {}

def get_tool_schemas() -> list[dict]:
    schemas = []
    for name, func in tools.items():
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        if not params:
            logger.warning(f"Tool '{name}' has no parameters, skipping schema generation.")
            continue

        param = params[0]
        annotation = param.annotation
        
        schemas.append({
            "name": name,
            "description": func.__doc__ or "Execute a quantum chemical calculation",
            "input_schema": annotation_to_json_schema(annotation)
        })
    return schemas

async def execute_tool(name: str, args: dict) -> Any:
    func = tools.get(name)
    if not func:
        return f"Error: Tool {name} not found."

    if not isinstance(args, dict):
        return f"Error: Tool {name} expects a dict of arguments, got {type(args).__name__}."

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    
    if not params:
        return f"Error: Tool {name} has no input parameters defined."

    annotation = params[0].annotation

    try:
        if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            data = annotation.model_validate(args)
        else:
            data = TypeAdapter(annotation).validate_python(args)
            
    except ValidationError as e:
        logger.warning(f"Validation failed for tool '{name}': {e}")
        return f"Invalid input for {name}: {e.json()}"
    except Exception as e:
        logger.exception(f"Unexpected error during validation of tool '{name}'")
        return f"Unexpected error during tool validation: {type(e).__name__}"

    try:
        if inspect.iscoroutinefunction(func):
            return await func(data)
        return func(data)
    except Exception as e:
        logger.exception(f"Tool '{name}' execution failed")
        return f"Error during execution of {name}: {str(e)}"