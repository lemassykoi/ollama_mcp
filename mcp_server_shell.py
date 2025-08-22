#!/usr/bin/env python3
"""
MCP Server pour exécuter des commandes shell locales avec retour complet et réessai possible.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pydantic import BaseModel, Field
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

logging.basicConfig(
    format="%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------------------
# Config
# -------------------------------
try:
    CURRENT_SHELL = os.environ.get("SHELL")
except Exception as e:
    logger.error(f"Unable to read current Shell: {e}")
    exit(1)

# Tool registry
tool_registry = {}
def register_tool(func):
    tool_registry[func.__name__] = func
    return func

# -------------------------------
# Lifespan
# -------------------------------
@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    yield {"shell": CURRENT_SHELL}

server = Server("MCP-SHELL", lifespan=server_lifespan)

# -------------------------------
# Input model
# -------------------------------
class RunCommandInput(BaseModel):
    cmd: list[str] = Field(..., description="Command as list of strings")
    timeout: int   = Field(10000, description="Timeout in milliseconds")

# -------------------------------
# Utils
# -------------------------------
async def run_shell_command(command: list[str], timeout_ms: int) -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode().strip(),
            "stderr": stderr.decode().strip()
        }
    except asyncio.TimeoutError:
        proc.kill()
        return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout_ms} ms"}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e)}

# -------------------------------
# MCP Tool
# -------------------------------
@register_tool
async def run_command(data: RunCommandInput) -> list[types.TextContent]:
    """
    Execute shell command with approval and timeout. Returns stdout, stderr, and returncode.
    """
    if not data.cmd:
        return [types.TextContent(type="text", text="No command provided.")]
    
    cmd_str = " ".join(data.cmd)

    lifespan = server.request_context.lifespan_context
    shell = lifespan["shell"]

    logger.info(f"Executing in shell {shell}: {cmd_str}")
    result = await run_shell_command(data.cmd, data.timeout)

    output = [
        types.TextContent(type="text", text=f"Command: {cmd_str}"),
        types.TextContent(type="text", text=f"Return code: {result['returncode']}"),
        types.TextContent(type="text", text=f"STDOUT:\n{result['stdout']}"),
        types.TextContent(type="text", text=f"STDERR:\n{result['stderr']}"),
    ]
    return output

run_command.input_model = RunCommandInput
run_command.input_schema = RunCommandInput.model_json_schema()

# -------------------------------
# Tool listing
# -------------------------------
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = []
    for name, func in tool_registry.items():
        tools.append(types.Tool(
            name=name,
            description=func.__doc__,
            inputSchema=getattr(func, "input_schema", {"type":"object","properties":{}})
        ))
    return tools

# -------------------------------
# call_tool handler
# -------------------------------
@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name not in tool_registry:
        raise ValueError(f"Tool not found: {name}")
    tool_func = tool_registry[name]
    input_model = getattr(tool_func, "input_model", None)
    try:
        data = input_model(**arguments) if input_model else None
        result = await tool_func(data) if data else await tool_func()
        return result if isinstance(result, list) else [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]

# -------------------------------
# Prompts (optionnel)
# -------------------------------
@server.list_prompts()
async def handle_list_prompts():
    return []

# -------------------------------
# Start MCP Server
# -------------------------------
async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="Shell MCP Server",
                server_version="0.1.0",
                instructions=f"You are a shell assistant. Current local shell is {CURRENT_SHELL}.",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(tools_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(run())
