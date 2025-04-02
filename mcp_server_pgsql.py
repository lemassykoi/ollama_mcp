import os
import mcp.server.stdio
import mcp.types as types
import urllib.parse
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
import re
import json
import asyncio
import asyncpg
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# POSTGRESQL INFOS
PG_USER = 'mcp_server'
PG_PASS = 'password'
PG_HOST = '10.0.0.1'
PG_PORT = 5432
PG_DB   = 'mcp_db'    # default DB to start with // user should have rights on the DB

DB_URI = f"postgres://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=disable"
PG_URI = f"postgres://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}"

POSTGRESQL_CONNECTION_KWARGS = {
    "autocommit": True,
    "row_factory": dict_row,
    "prepare_threshold": 0,
}

READ_ONLY_REGEX = re.compile(r'^\s*SELECT\b', re.IGNORECASE)

# Global variables for state management
active_db = None
db_pools = {}
conn_global = None

# Tool registry
tool_registry = {}

def register_tool(tool_func):
    tool_registry[tool_func.__name__] = tool_func
    return tool_func

@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    global conn_global, active_db, db_pools
    # Set the default active database (e.g., "postgres")
    active_db = os.environ.get("DEFAULT_DB", PG_DB)
    # Get the base connection string (pointing to an administrative database)
    database_url = os.environ.get("DATABASE_URL", DB_URI)

    # Create a global connection for admin tasks if needed
    conn_global = await asyncpg.connect(database_url)
    # Create a connection pool for the default active database
    default_pool = await asyncpg.create_pool(dsn=database_url)
    db_pools[active_db] = default_pool

    try:
        # Yield the lifespan context containing state
        yield {"conn": conn_global, "active_db": active_db, "db_pools": db_pools}
    finally:
        await conn_global.close()
        # Close all connection pools
        for pool in db_pools.values():
            await pool.close()
        conn_global = None

# Create a server instance
server = Server(
    "IPC-MCP-SERVER",
    lifespan=server_lifespan
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

# -------------------------------
# Pydantic models
# -------------------------------
class SwitchDBInput(BaseModel):
    target_db: str = Field(..., description="Target database name.")

class RunSelectQueryInput(BaseModel):
    query: str = Field(..., description="The SELECT query to execute.")

class ListTablesQueryInput(BaseModel):
    db_name: str = Field(..., description="The database we will list tables from.")

# -------------------------------
# Tools Functions
# -------------------------------
@register_tool
async def list_tables(db_name: str) -> str:
    """
    Lists all available tables in the specified database.
    """
    logger.info('Run tool list_tables')
    lifespan = server.request_context.lifespan_context
    lifespan["active_db"] = db_name
    db_pools = lifespan["db_pools"]

    if db_name not in db_pools:
        base_db_url = os.environ.get("POSTGRESQL_URL", PG_URI)
        target_db_url = update_db_in_url(base_db_url, db_name)
        try:
            db_pools[db_name] = await asyncpg.create_pool(dsn=target_db_url)
            logger.info(f"DB switched to {db_name}")
        except Exception as e:
            raise ValueError(json.dumps(str(e)))

    pool = lifespan["db_pools"][db_name]
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema') AND table_type = 'BASE TABLE';")
    tables = [row["table_name"] for row in rows]
    return json.dumps(tables)

list_tables.input_model = ListTablesQueryInput
list_tables.input_schema = ListTablesQueryInput.model_json_schema()

@register_tool
async def run_select_query(data: RunSelectQueryInput) -> str:
    """
    Executes a user-supplied SQL query on the active PostgreSQL database. Only queries that begin with SELECT (read-only) are allowed.
    """
    if not is_read_only(data.query):
        raise ValueError("Only SELECT queries are allowed with this tool.")

    lifespan = server.request_context.lifespan_context
    current_db = lifespan["active_db"]
    pool = lifespan["db_pools"][current_db]

    async with pool.acquire() as conn:
        rows = await conn.fetch(data.query)
        result = [dict(row) for row in rows]

    return json.dumps(result)

run_select_query.input_model = RunSelectQueryInput
run_select_query.input_schema = RunSelectQueryInput.model_json_schema()

@register_tool
async def switch_db(data: SwitchDBInput) -> str:
    """
    Switches the active database in the lifespan context to the specified target database.
    If a connection pool for the target database does not exist, it creates one.
    """
    lifespan = server.request_context.lifespan_context
    lifespan["active_db"] = data.target_db
    db_pools = lifespan["db_pools"]

    if data.target_db not in db_pools:
        base_db_url = os.environ.get("POSTGRESQL_URL", PG_URI)
        target_db_url = update_db_in_url(base_db_url, data.target_db)
        try:
            db_pools[data.target_db] = await asyncpg.create_pool(dsn=target_db_url)
        except Exception as e:
            error_msg = f"Error switching active database: {str(e)}"
            raise ValueError(error_msg)

    return json.dumps(f"Switched active database to '{data.target_db}'.")

switch_db.input_model = SwitchDBInput
switch_db.input_schema = SwitchDBInput.model_json_schema()


@register_tool
async def get_current_active_db() -> str:
    """
    Returns the currently active database from the lifespan context.
    """
    lifespan = server.request_context.lifespan_context
    active_db = lifespan["active_db"]
    logger.info(f"Active DB: {active_db}")
    return active_db

@register_tool
async def get_server_info() -> str:
    """
    Retrieves information about the connected PostgreSQL server.
    """
    conn = server.request_context.lifespan_context["conn"]
    row = await conn.fetchrow("""
        SELECT
            version() AS pg_version,
            current_database() AS current_db,
            inet_server_addr() AS server_ip,
            inet_server_port() AS server_port,
            pg_postmaster_start_time() AS start_time
    """)
    return json.dumps({
        "PostgreSQL Version": str(row["pg_version"]),
        "Current Database": str(row["current_db"]),
        "Server IP": str(row["server_ip"]),
        "Server Port": str(row["server_port"]),
        "Uptime Since": str(row["start_time"])
    })

@register_tool
async def list_databases() -> str:
    """
    Lists all available databases in the PostgreSQL instance.
    """
    logger.info('Tool List Databases')
    lifespan = server.request_context.lifespan_context
    current_db = lifespan["active_db"]
    logger.info(f'Active DB: {current_db}')
    pool = lifespan["db_pools"][current_db]

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT datname FROM pg_database WHERE datistemplate = false;")
    databases = [row["datname"] for row in rows]
    return databases

# -------------------------------
# Helper Functions
# -------------------------------

def is_read_only(query: str) -> bool:
    """Simple check to ensure the query starts with SELECT."""
    return READ_ONLY_REGEX.match(query) is not None

def update_db_in_url(url: str, new_db: str) -> str:
    """
    Update the database name in a PostgreSQL connection URL.
    """
    parsed = urllib.parse.urlparse(url)
    new_path = f"/{new_db}?sslmode=disable"
    new_url = parsed._replace(path=new_path)
    return urllib.parse.urlunparse(new_url)

# -------------------------------
# Start MCP Server
# -------------------------------

async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="IPC SQL MCP Server",
                server_version="0.1.0",
                instructions="You are a senior SQL Manager, with given access to local PostgreSQL Server.",
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
