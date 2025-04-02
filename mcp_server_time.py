import pytz
import datetime
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from typing import Literal
from dateutil import parser as dateutil_parser
from pydantic import BaseModel, Field
import asyncio
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create a server instance
server = Server(
    "IPC MCP TIME SERVER",
)

# Tool registry
tool_registry = {}

def register_tool(tool_func):
    tool_registry[tool_func.__name__] = tool_func
    return tool_func

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    tools = []
    for name, func in tool_registry.items():
        tools.append(types.Tool(
            name=name,
            description=func.__doc__,
            inputSchema=getattr(func, 'input_schema', {
                "type": "object",
                "properties": {},
                "required": []
            })
        ))
    return tools

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="example-prompt",
            description="An example prompt template",
            arguments=[
                types.PromptArgument(
                    name="arg1", description="Example Prompt argument", required=True
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name != "example-prompt":
        raise ValueError(f"Unknown prompt: {name}")

    return types.GetPromptResult(
        description="Example prompt",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text="Example prompt text"),
            )
        ],
    )

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name in tool_registry:
        try:
            tool_func = tool_registry[name]
            input_model = getattr(tool_func, 'input_model', None)
            if input_model:
                data = input_model(**arguments)
                result = await tool_func(data)
            else:
                result = await tool_func()
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=str(e))]
    raise ValueError(f"Tool not found: {name}")

## PYDANTIC Models
class FormatTimeInput(BaseModel):
    format: str = Field(
        "%Y-%m-%d %H:%M:%S", description="Python strftime format string"
    )
    timezone: str = Field(
        "UTC", description="IANA timezone name (e.g., UTC, America/New_York)"
    )

class ConvertTimeInput(BaseModel):
    timestamp: str = Field(
        ..., description="ISO 8601 formatted time string (e.g., 2024-01-01T12:00:00Z)"
    )
    from_tz: str = Field(
        ..., description="Original IANA time zone of input (e.g. UTC or Europe/Berlin)"
    )
    to_tz: str = Field(..., description="Target IANA time zone to convert to")

class ElapsedTimeInput(BaseModel):
    start: str = Field(..., description="Start timestamp in ISO 8601 format")
    end: str = Field(..., description="End timestamp in ISO 8601 format")
    units: Literal["seconds", "minutes", "hours", "days"] = Field(
        "seconds", description="Unit for elapsed time"
    )

class ParseTimestampInput(BaseModel):
    timestamp: str = Field(
        ..., description="Flexible input timestamp string (e.g., 2024-06-01 12:00 PM)"
    )
    timezone: str = Field(
        "UTC", description="Assumed timezone if none is specified in input"
    )

## Tools Functions

@register_tool
async def get_current_utc() -> str:
    """
    Returns the current time in UTC in ISO format.
    """
    try:
        utc_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=datetime.timezone.utc).isoformat()
        return {"utc": utc_time}
    except Exception as e:
        return f"An error occurred when fetching time: {str(e)}"

@register_tool
async def get_current_local() -> str:
    """
    Returns the current time in local timezone in ISO format.
    """
    try:
        local_time = datetime.datetime.now().isoformat()
        return {"local_time": local_time}
    except Exception as e:
        return f"An error occurred when fetching time: {str(e)}"

@register_tool
async def format_current_time(data: FormatTimeInput) -> str:
    """
    Return the current time formatted for a specific timezone and format.
    """
    try:
        tz = pytz.timezone(data.timezone)
    except Exception:
        return f"Invalid timezone: {data.timezone}"

    now = datetime.datetime.now(tz)

    try:
        return {"formatted_time": now.strftime(data.format)}
    except Exception as e:
        return f"Invalid format string: {e}"

format_current_time.input_model = FormatTimeInput
format_current_time.input_schema = FormatTimeInput.model_json_schema()

@register_tool
async def convert_time(data: ConvertTimeInput) -> str:
    """
    Convert a timestamp from one timezone to another.
    """
    try:
        from_zone = pytz.timezone(data.from_tz)
        to_zone = pytz.timezone(data.to_tz)
    except Exception as e:
        return f"Invalid timezone: {e}"

    try:
        dt = dateutil_parser.parse(data.timestamp)
        if dt.tzinfo is None:
            dt = from_zone.localize(dt)
        else:
            dt = dt.astimezone(from_zone)
        converted = dt.astimezone(to_zone)
        return {"converted_time": converted.isoformat()}
    except Exception as e:
        return f"Invalid timestamp: {e}"

convert_time.input_model = ConvertTimeInput
convert_time.input_schema = ConvertTimeInput.model_json_schema()

@register_tool
async def elapsed_time(data: ElapsedTimeInput) -> str:
    """
    Calculate the difference between two timestamps in chosen units.
    """
    try:
        start_dt = dateutil_parser.parse(data.start)
        end_dt = dateutil_parser.parse(data.end)
        delta = end_dt - start_dt
    except Exception as e:
        return f"Invalid timestamps: {e}"

    seconds = delta.total_seconds()
    result = {
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
        "days": seconds / 86400,
    }

    return {"elapsed": result[data.units], "unit": data.units}

elapsed_time.input_model = ElapsedTimeInput
elapsed_time.input_schema = ElapsedTimeInput.model_json_schema()

@register_tool
async def parse_timestamp(data: ParseTimestampInput) -> str:
    """
    Parse human-friendly input timestamp and return standardized UTC ISO time.
    """
    try:
        tz = pytz.timezone(data.timezone)
        dt = dateutil_parser.parse(data.timestamp)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        dt_utc = dt.astimezone(pytz.utc)
        return {"utc": dt_utc.isoformat()}
    except Exception as e:
        return f"Could not parse: {e}"

parse_timestamp.input_model = ParseTimestampInput
parse_timestamp.input_schema = ParseTimestampInput.model_json_schema()

## Start MCP Server
async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="IPC Time MCP Server",
                server_version="0.1.0",
                instructions="You are a Time Server, kind of an advanced Speaking Clock, with access to realtime clock.",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(
                        tools_changed=True
                    ),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(run())
