import asyncio
import shlex
from typing import List, Optional, Dict, Any, Mapping
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Resource, Tool
import requests
import json
import colorama
import os
import logging
import urllib.parse

B_RED        = colorama.Back.RED
RED          = colorama.Fore.RED
BLUE         = colorama.Fore.BLUE
CYAN         = colorama.Fore.CYAN
GREEN        = colorama.Fore.GREEN
YELLOW       = colorama.Fore.YELLOW
MAGENTA      = colorama.Fore.MAGENTA
YELLOW_LIGHT = colorama.Fore.LIGHTYELLOW_EX
RESET        = colorama.Style.RESET_ALL

## MODELS
model = "llama3-groq-tool-use:8b-q8_0"
#model = "qwen2.5:14b-instruct"
#model = "qwq:32b-q4_K_M"
#model = "mistral-small:latest"

# Global settings for Ollama API calls
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_SEED: int = 1234567890

# Global URL and model settings for the Ollama API.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

# Initialize logger
logging.basicConfig(
    format = '%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)',
    level  = logging.INFO
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Helper function to build a resource list (from MCP resources)
def build_resource_list(resources: List[Resource]) -> List[Dict[str, str]]:
    resource_list = []
    for resource in resources:
        if resource[0] == 'resources':
            for item in resource[1]:
                logger.debug(item)
                resource_dict = {
                    'name': item.name,
                    'uri': str(item.uri)  # Convert AnyUrl to string
                }
                resource_list.append(resource_dict)
    return resource_list

class MultiServerMCPClient:
    """
    A multi-server MCP client that supports connecting to multiple MCP servers,
    merging their tool lists (with namespaced tool names), and dispatching tool calls
    to the appropriate server.
    """
    def __init__(self, servers_to_connect:dict, history: list[Mapping[str, Any]] = []) -> None:
        logger.info("Multi Server MCP Client v3 initialization...")

        self.sessions: Dict[str, ClientSession] = {}          # Persistent sessions keyed by server name.
        self.exit_stack = AsyncExitStack()                    # For proper async cleanup.
        self.ollama_api_url = OLLAMA_BASE_URL
        self.tools: Dict[str, List[Tool]] = {}                # Tools provided by each server.
        self.resources: Dict[str, List[Dict[str, str]]] = {}  # Resources from each server.
        self.servers_to_connect: Dict = servers_to_connect
        self.first_key = next(iter(servers_to_connect))
        self.connections = []
        self.merged_ollama_tools = []
        self.merged_tools_list_names = []
        self.merged_resources_list_names = []
        self.history = history

    async def gather_connections(self):
        for server_name, script_path in self.servers_to_connect.items():
            if os.path.isfile(script_path):  # Connect to a local MCP server
                connection = self.connect_to_server(server_name, script_path)
            else:  # Connect to a custom server
                connection = self.connect_to_custom_server(server_name, script_path)
            self.connections.append(connection)
        await asyncio.gather(*self.connections)

        for server_name in self.sessions:
            # Convert and namespace tools.
            ollama_tool_list = self.convert_mcp_tools_to_ollama_tools(self.tools[server_name], server_name)
            tool_names = [f"{server_name}:tool:{tool.name}" for tool in self.tools[server_name]]

            # Convert resources to callable tools.
            ollama_resource_list = self.convert_mcp_resources_to_ollama_tools(self.resources[server_name], server_name)
            resource_names = [f"{server_name}:resource:{res['name']}" for res in self.resources[server_name]]

            # Build merged list
            self.merged_ollama_tools += ollama_tool_list + ollama_resource_list
            self.merged_tools_list_names += tool_names + resource_names
            self.merged_resources_list_names += resource_names

        logger.info("Multi Server MCP Client v3 fully initialized.")

    def convert_mcp_tools_to_ollama_tools(self, tools: List[Tool], server_name: str) -> List[Dict]:
        """Convert MCP tools to Ollama tools with explicit call type 'tool'."""
        ollama_tools = []
        for tool in tools:
            parameters = {
                "type": "object",
                "properties": {},
                "required": tool.inputSchema.get("required", [])
            }

            for param_name, param_info in tool.inputSchema["properties"].items():
                if "anyOf" in param_info:
                    # Handle 'anyOf' field
                    anyOf_schemas = param_info["anyOf"]
                    for schema in anyOf_schemas:
                        if schema["type"] == "array":
                            parameters["properties"][param_name] = {
                                "type": "array",
                                "items": {"type": schema["items"]["type"]},
                                "description": param_info.get("description", "").strip()
                            }
                        elif schema["type"] == "null":
                            # Mark parameter as nullable.
                            if param_name in parameters["properties"]:
                                parameters["properties"][param_name]["nullable"] = True
                else:
                    parameters["properties"][param_name] = {
                        "type": param_info["type"],
                        "description": param_info.get("description", "").strip()
                    }

            ollama_tools.append({
                'type': 'function',
                'function': {
                    'name': f'{server_name}:tool:{tool.name}',
                    'description': tool.description.strip(),
                    'parameters': parameters
                }
            })
        return ollama_tools

    def convert_mcp_resources_to_ollama_tools(self, resources: List[Dict[str, str]], server_name: str) -> List[Dict]:
        """Expose MCP resources as Ollama callable tools with explicit call type 'resource'."""
        ollama_resource_tools = []
        for resource in resources:
            parameters = {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource"
                    }
                },
                "required": ["uri"]
            }
            ollama_resource_tools.append({
                'type': 'function',
                'function': {
                    'name': f'{server_name}:resource:{resource["name"]}',
                    'description': f"Access the resource {resource['name']}",
                    'parameters': parameters
                }
            })
        return ollama_resource_tools

    async def connect_to_server(self, server_name: str, server_script_path: str) -> None:
        """
        Connect to an MCP server using a script (e.g., a .py or .js file) and store the session under server_name.
        """
        # Determine the command based on file extension.
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        self.sessions[server_name] = session

        # Retrieve resources and tools.
        try:
            resources_raw = await session.list_resources()
        except Exception:
            logger.warning(f"list_resources not supported by {server_name}")
            resources_raw = {"result": []}

        try:
            tools_raw = await session.list_tools()
        except Exception:
            logger.warning(f"list_tools not supported by {server_name}")
            tools_raw = {"tools": []}

        self.tools[server_name] = tools_raw.tools
        self.resources[server_name] = build_resource_list(resources_raw)
        if tools_raw != []:
            print(f"\nConnected to server '{server_name}' with tools:",
                [tool.name for tool in tools_raw.tools])
        try:
            print(f"\nConnected to server '{server_name}' with resources:",
                [resource.name for resource in resources_raw.resources])
        except AttributeError:
            print(f"\nNo Resources to Connect for Server '{server_name}'")

    async def connect_to_custom_server(self, server_name: str, command_line: str) -> None:
        """
        Connect to an MCP server using a custom command line.
        For example, for a prebuilt server launched with:
          npx -y @executeautomation/playwright-mcp-server
        or, in your case, the filesystem server:
          npx -y @modelcontextprotocol/server-filesystem /home/clement
        """
        # Use shlex to split the command line into command and args.
        parts = shlex.split(command_line)
        if not parts:
            raise ValueError("Invalid command line provided.")
        command = parts[0]
        args = parts[1:]
        server_params = StdioServerParameters(
            command = command,
            args    = args,
            env     = None
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        self.sessions[server_name] = session

        # Retrieve resources and tools.
        try:
            resources_raw = await session.list_resources()
        except Exception as e:
            logger.warning(f"list_resources not supported by {server_name}: {e}")
            resources_raw = {"result": []}
        try:
            tools_raw = await session.list_tools()
        except Exception as e:
            logger.warning(f"list_tools not supported by {server_name}: {e}")
            tools_raw = {"tools": []}

        self.tools[server_name] = tools_raw.get("tools", []) if isinstance(tools_raw, dict) else tools_raw.tools
        self.resources[server_name] = build_resource_list(resources_raw)
        print(f"\nConnected to custom server '{server_name}' with tools:",
              [tool.name for tool in self.tools[server_name]])

    async def call_tool(self, server_name: str, tool_name: str, arguments: str):
        """
        Call a tool on a specified server.
        """
        session = self.sessions.get(server_name)
        if session is None:
            raise ValueError(f"Server '{server_name}' not connected")
        return await session.call_tool(tool_name, arguments)

    async def read_resource(self, server_name: str, uri:str):  ## have to be called from process_query()
        """
        Read Resource on a specified server.
        """
        session = self.sessions.get(server_name)
        if session is None:
            raise ValueError(f"Server '{server_name}' not connected")
        meta, mime_type = await session.read_resource(uri)
        logger.debug(str(mime_type)[:256])
        return (urllib.parse.unquote(mime_type[1][-1].blob))

    async def process_query(
        self,
        query: str = "",
        messages: Optional[List[Dict]] = None,
        tool_call_messages = [],
    ) -> tuple[str, list[dict]]:
        """
        Process a query by sending it to the Ollama API with the merged tool list.
        When a tool call is returned, the function examines the tool name's prefix and
        dispatches the call to the appropriate server based on the explicit call type.
        """
        if query and messages is None:  ## first message : user query is present, but messages is None
            self.history.append({"role": "user", "content": query})

        elif messages is not None:   ## next messages : adding user query to messages history
            if query != "":
                self.history.append({"role": "user", "content": query})

        elif not query and messages is None:  ## no user query and no messages => this is a tool call
            logger.debug(tool_call_messages)

        else:
            logger.error('PROBLEM')
            exit(1)


        logger.debug(f'Sending query with tools: {self.merged_ollama_tools}')
        logger.info(f'Sending query with tool names:\n{[tool["function"]["name"] for tool in self.merged_ollama_tools]}')

        # Prepare the payload for Ollama.
        data = {
            "model": model,
            "messages": self.history,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                'temperature': OLLAMA_TEMPERATURE,
                'seed': OLLAMA_SEED
            },
            "tools": self.merged_ollama_tools,
        }
        payload = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        # Send the initial query to Ollama.
        response = requests.post(OLLAMA_CHAT_URL, headers=headers, data=payload, stream=False)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(response_data)

        message = response_data.get('message')
        tool_calls = message.get('tool_calls')

        if tool_calls:
            logger.info(f'  => Tool Calls : {len(tool_calls)}')

            for tool_call in tool_calls:
                inner_dict = tool_call['function']
                full_function_name = inner_dict['name']  # e.g. "ttsserver:tool:play_audio" or "ttsserver:resource:audio_file"
                logger.info(f"Call for {full_function_name}")
                arguments = inner_dict['arguments']
                logger.info(f"With Args: {arguments}")

                # Extract the server prefix, call type, and name.
                parts = full_function_name.split(':', 2)
                if len(parts) != 3:
                    logger.error("Fallback to a default as not all parts are present.")
                    server_id, call_type, name = self.default_server, 'tool', full_function_name
                else:
                    server_id, call_type, name = parts
                logger.info(f"Check for {name}")

                # Check if the full name exists in the merged tools list.
                if full_function_name in self.merged_tools_list_names:
                    logger.info(f"Dispatching {call_type} call to server '{server_id}' for '{name}'")
                    session = self.sessions.get(server_id)

                    if session is None:
                        logger.error(f"Server '{server_id}' not connected.")
                        answer = json.dumps({"success": False, "error": f"Server '{server_id}' not connected."})

                    else:
                        if call_type == "tool":
                            logger.info('Call Type: Tool')
                            result = await session.call_tool(name, arguments)
                            logger.info(result)

                            if result.isError:
                                logger.error('Tool Call Error')
                                answer = json.dumps({"success": False, "error": message})

                            else:
                                logger.info('Tool Call Success')
                                if result.content[0].text:
                                    logger.info('Tool result is Text')
                                    message = result.content[0].text
                                else:
                                    logger.warning('UNKNOWN Tool Result Content')
                                    message = str(result.content[0])
                                
                                answer = json.dumps({"success": True, "message": message})

                        elif call_type == "resource":
                            logger.info('Call Type: Resource')
                            try:
                                args_dict = json.loads(arguments)
                            except Exception:
                                args_dict = arguments
                            uri = args_dict.get("uri")

                            if not uri:
                                logger.error("URI parameter missing for resource call")
                                answer = json.dumps({"success": False, "error": "URI parameter missing for resource call."})

                            else:
                                meta, resource_message = await session.read_resource(uri)
                                if resource_message[1][-1].blob:
                                    try:
                                        answer = json.dumps({"success": True, "message": urllib.parse.unquote(resource_message[1][-1].blob)})
                                    except Exception as e:
                                        answer = {"success": True, "message": urllib.parse.unquote(resource_message[1][-1].blob)}

                                elif resource_message[1][-1].text:
                                    try:
                                        answer = json.dumps({"success": True, "message": urllib.parse.unquote(resource_message[1][-1].text)})
                                    except Exception as e:
                                        answer = {"success": True, "message": urllib.parse.unquote(resource_message[1][-1].text)}

                                else:
                                    logger.error('UNKNOWN Resource Response')
                                    logger.info(resource_message[1][-1][:256])

                        else:
                            logger.error(f"Unknown call type: {call_type}")
                            answer = json.dumps({"success": False, "error": f"Unknown call type: {call_type}"})
                else:
                    logger.error('Function not found in merged tool list')
                    answer = json.dumps({"success": False, "error": "Invalid function name."})

                ## Append tool call result to messages
                try:
                    self.history.append({"role": "tool", "content": json.dumps(answer)})
                    tool_call_messages = {"role": "tool", "content": json.dumps(answer)}

                except json.JSONDecodeError:
                    self.history.append({"role": "tool", "content": answer})
                    tool_call_messages = {"role": "tool", "content": answer}

            return await self.process_query(
                tool_call_messages = tool_call_messages,
            )

        self.history.append(response_data['message'])
        return self.history

    async def cleanup(self):
        """
        Clean up all connections before exit.
        """
        try:
            await self.exit_stack.aclose()
        except RuntimeError as e:
            if "Attempted to exit cancel scope" in str(e):
                logger.warning("Cancellation scope error during cleanup; ignoring.")
            else:
                logger.error(str(e))
