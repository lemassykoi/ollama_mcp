import os
import pathlib
import asyncio
import difflib
import logging
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

logging.basicConfig(
    format='%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_DIRECTORIES = [
    str(pathlib.Path(os.path.expanduser("/mnt/NAS")).resolve()),     ## CHANGE WITH YOUR OWN DIRECTORIES
    str(pathlib.Path(os.path.expanduser("~/scripts")).resolve()),    ## CHANGE WITH YOUR OWN DIRECTORIES
]

# Tool registry
tool_registry = {}

def register_tool(tool_func):
    tool_registry[tool_func.__name__] = tool_func
    return tool_func

# Create a server instance
server = Server(
    "FILESYSTEM-MCP-SERVER",
)

@server.list_tools()
async def list_tools() -> list[types.Tool]:
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

# Access lifespan context in handlers
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

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def normalize_path(requested_path: str) -> pathlib.Path:
    requested = pathlib.Path(os.path.expanduser(requested_path)).resolve()
    for allowed in ALLOWED_DIRECTORIES:
        if str(requested).startswith(allowed):
            return requested
    raise ValueError(f"Access denied: {requested} is outside allowed directories.")

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------

class ReadFileRequest(BaseModel):
    path: str = Field(..., description="Path to the file to read")

class WriteFileRequest(BaseModel):
    path: str = Field(..., description="Path to write to. Existing file will be overwritten.")
    content: str = Field(..., description="UTF-8 encoded text content to write.")

class EditOperation(BaseModel):
    oldText: str = Field(..., description="Text to find and replace (exact match required)")
    newText: str = Field(..., description="Replacement text")

class EditFileRequest(BaseModel):
    path: str = Field(..., description="Path to the file to edit.")
    edits: List[EditOperation] = Field(..., description="List of edits to apply.")
    dryRun: bool = Field(False, description="If true, only return diff without modifying file.")

class CreateDirectoryRequest(BaseModel):
    path: str = Field(..., description="Directory path to create. Intermediate dirs are created automatically.")

class ListDirectoryRequest(BaseModel):
    path: str = Field(..., description="Directory path to list contents for.")

class DirectoryTreeRequest(BaseModel):
    path: str = Field(..., description="Directory path for which to return recursive tree.")

class SearchFilesRequest(BaseModel):
    path: str = Field(..., description="Base directory to search in.")
    pattern: str = Field(..., description="Filename pattern (case-insensitive substring match).")
    excludePatterns: Optional[List[str]] = Field(default=[], description="Patterns to exclude.")

# ------------------------------------------------------------------------------
# Tools Functions
# ------------------------------------------------------------------------------

@register_tool
async def read_file(data: ReadFileRequest) -> str:
    """
    Read the entire contents of a file.
    """
    path = normalize_path(data.path)
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(str(e))

read_file.input_model = ReadFileRequest
read_file.input_schema = ReadFileRequest.model_json_schema()

@register_tool
async def write_file(data: WriteFileRequest) -> str:
    """
    Write content to a file, overwriting if it exists.
    """
    path = normalize_path(data.path)
    try:
        path.write_text(data.content, encoding="utf-8")
        return f"Successfully wrote to {data.path}"
    except Exception as e:
        raise ValueError(str(e))

write_file.input_model = WriteFileRequest
write_file.input_schema = WriteFileRequest.model_json_schema()

@register_tool
async def edit_file(data: EditFileRequest) -> str:
    """
    Apply a list of edits to a text file. Support dry-run to get unified diff.
    """
    path = normalize_path(data.path)
    original = path.read_text(encoding="utf-8")
    modified = original

    for edit in data.edits:
        if edit.oldText not in modified:
            raise ValueError(f"oldText not found in content: {edit.oldText[:50]}")
        modified = modified.replace(edit.oldText, edit.newText, 1)

    if data.dryRun:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="original",
            tofile="modified",
        )
        return "".join(diff)

    path.write_text(modified, encoding="utf-8")
    return f"Successfully edited file {data.path}"

edit_file.input_model = EditFileRequest
edit_file.input_schema = EditFileRequest.model_json_schema()

@register_tool
async def create_directory(data: CreateDirectoryRequest) -> str:
    """
    Create a new directory recursively.
    """
    dir_path = normalize_path(data.path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory {data.path}"
    except Exception as e:
        raise ValueError(str(e))

create_directory.input_model = CreateDirectoryRequest
create_directory.input_schema = CreateDirectoryRequest.model_json_schema()

@register_tool
async def list_directory(data: ListDirectoryRequest) -> str:
    """
    List contents of a directory.
    """
    dir_path = normalize_path(data.path)
    if not dir_path.is_dir():
        raise ValueError("Provided path is not a directory")

    listing = []
    for entry in dir_path.iterdir():
        prefix = "[DIR]" if entry.is_dir() else "[FILE]"
        listing.append(f"{prefix} {entry.name}")

    return "\n".join(listing)

list_directory.input_model = ListDirectoryRequest
list_directory.input_schema = ListDirectoryRequest.model_json_schema()

@register_tool
async def directory_tree(data: DirectoryTreeRequest) -> dict:
    """
    Recursively return a tree structure of a directory.
    """
    base_path = normalize_path(data.path)

    def build_tree(current: pathlib.Path):
        entries = []
        for item in current.iterdir():
            entry = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
            }
            if item.is_dir():
                entry["children"] = build_tree(item)
            entries.append(entry)
        return entries

    return build_tree(base_path)

directory_tree.input_model = DirectoryTreeRequest
directory_tree.input_schema = DirectoryTreeRequest.model_json_schema()

@register_tool
async def search_files(data: SearchFilesRequest) -> dict:
    """
    Search files and directories matching a pattern.
    """
    base_path = normalize_path(data.path)
    results = []

    for root, dirs, files in os.walk(base_path):
        root_path = pathlib.Path(root)
        # Apply exclusion patterns
        excluded = False
        for pattern in data.excludePatterns:
            if pathlib.Path(root).match(pattern):
                excluded = True
                break
        if excluded:
            continue
        for item in files + dirs:
            if data.pattern.lower() in item.lower():
                result_path = root_path / item
                if any(str(result_path).startswith(alt) for alt in ALLOWED_DIRECTORIES):
                    results.append(str(result_path))

    return {"matches": results or ["No matches found"]}

search_files.input_model = SearchFilesRequest
search_files.input_schema = SearchFilesRequest.model_json_schema()

@register_tool
async def list_allowed_directories() -> dict:
    """
    Show all directories this server can access.
    """
    return {"allowed_directories": ALLOWED_DIRECTORIES}

# ------------------------------------------------------------------------------
# Start MCP Server
# ------------------------------------------------------------------------------

async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="IPC FILESYSTEM MCP Server",
                server_version="0.1.0",
                instructions="You are a filesystem manager, with given access to specific directories.",
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
