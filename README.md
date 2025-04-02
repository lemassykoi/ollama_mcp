 # Multi-Server MCP Ollama Client Documentation

This documentation provides an overview of the functionalities and usage of two Python scripts, `mcp_client_v2.py` and `main.py`. These scripts are components of a larger system that allows for communication with multiple servers using the Model Context Protocol (MCP). The main script is `main.py`, which uses the `MultiServerMCPClient` class from `mcp_client_v2.py` to establish connections, process queries, and manage server interactions.

## Table of Contents
1. [Introduction](#introduction)
2. [`mcp_client_v2.py`](#mcp_client_v2py)
    - [Overview](#overview)
    - [Classes and Methods](#classes-and-methods)
        * [MultiServerMCPClient Class](#multiservermcpcclient-class)
            + [`__init__` Method](#init-method)
            + [`gather_connections` Method](#gather_connections-method)
            + [`convert_mcp_tools_to_ollama_tools` Method](#convert_mcp_tools_to_ollama_tools-method)
            + [`convert_mcp_resources_to_ollama_tools` Method](#convert_mcp_resources_to_ollama_tools-method)
            + [`connect_to_server` Method](#connect_to_server-method)
            + [`connect_to_custom_server` Method](#connect_to_custom_server-method)
            + [`call_tool` Method](#call_tool-method)
            + [`read_resource` Method](#read_resource-method)
            + [`process_query` Method](#process_query-method)
            + [`cleanup` Method](#cleanup-method)
3. [`main.py`](#mainpy)
    - [Overview](#overview-1)
    - [Main Function](#main-function)
4. [Usage](#usage)
5. [Conclusion](#conclusion)

## Introduction

The MCP is a JSON-RPC based protocol that enables models to communicate with tools and resources in an extensible and interoperable manner. The `mcp_client_v2.py` script provides the implementation of a multi-server MCP client that supports connecting to multiple MCP servers, merging their tool lists (with namespaced tool names), and dispatching tool calls to the appropriate server based on the explicit call type.

The `main.py` script is an example usage of the `MultiServerMCPClient` class from `mcp_client_v2.py`. It demonstrates how to establish connections with multiple servers, process queries, and manage interactions using this client.

## `mcp_client_v2.py`

### Overview

The `mcp_client_v2.py` script contains the implementation of a multi-server MCP client that allows communication with various MCP servers, including local servers and custom servers. The main class, `MultiServerMCPClient`, provides methods for connecting to multiple servers, processing queries, and dispatching tool calls to the appropriate server based on the call type (tool or resource).

### Classes and Methods

#### MultiServerMCPClient Class

- **`__init__(self, servers_to_connect: dict, history: list[Mapping[str, Any]] = []) -> None:`**
    - Initializes the `MultiServerMCPClient` object with the given servers to connect and an optional message history. Sets up logging and initializes necessary variables for server connections, tools, resources, and Ollama API calls.
- **`gather_connections(self)`**
    - Establishes connections to all specified servers using their script paths or custom command lines. Retrieves the tool and resource lists from each connected server and merges them into a single list for use with the Ollama API.
- **`convert_mcp_tools_to_ollama_tools(self, tools: List[Tool], server_name: str) -> List[Dict]:`**
    - Converts MCP tools to Ollama tools with an explicit call type 'tool'. This method is used to prepare the tool list for use with the Ollama API.
- **`convert_mcp_resources_to_ollama_tools(self, resources: List[Dict[str, str]], server_name: str) -> List[Dict]:`**
    - Exposes MCP resources as Ollama callable tools with an explicit call type 'resource'. This method is used to make resources accessible through the Ollama API.
- **`connect_to_server(self, server_name: str, server_script_path: str) -> None:`**
    - Connects to an MCP server using a script (either Python or JavaScript) and stores the session under the specified server name. Retrieves resources and tools from the connected server and logs their names.
- **`connect_to_custom_server(self, server_name: str, command_line: str) -> None:`**
    - Connects to an MCP server using a custom command line. This method allows for connecting to prebuilt servers or custom servers launched with specific commands. Retrieves resources and tools from the connected server and logs their names.
- **`call_tool(self, server_name: str, tool_name: str, arguments: str):`**
    - Calls a tool on a specified server using the given tool name and arguments. Returns the result of the tool call.
- **`read_resource(self, server_name: str, uri:str) -> Any:`**
    - Reads a resource from a specified server using the provided URI. Returns the content of the resource.
- **`process_query(self, query: str = "", messages: Optional[List[Dict]] = None, tool_call_messages = []) -> tuple[str, list[dict]]:`**
    - Processes a query by sending it to the Ollama API with the merged tool list. When a tool call is returned, this method examines the tool name's prefix and dispatches the call to the appropriate server based on the explicit call type (tool or resource). Returns the updated message history along with the result of any tool calls.
- **`cleanup(self)`**
    - Cleans up all connections before exit. This method ensures that resources are properly released when the client is no longer in use.

## `main.py`

### Overview

The `main.py` script demonstrates how to use the `MultiServerMCPClient` class from `mcp_client_v2.py` to establish connections with multiple servers, process queries, and manage interactions. The main function sets up a dictionary of servers to connect to and creates an instance of the `MultiServerMCPClient` class using this dictionary. It then enters a loop where it continuously takes user input, processes queries using the client, and prints the responses.

### Main Function

- **`main()`**
    - Sets up a dictionary of servers to connect to. (edit `servers_to_connect` in `main.py` if needed)
    - Creates an instance of the `MultiServerMCPClient` class using this server list.
    - Enters a loop where it continuously takes user input, processes queries using the client, and prints the responses.
    - Cleans up connections when the user exits the loop or encounters errors.

## Usage

To use these scripts, follow the steps below:

1. Ensure that you have Python 3 installed on your system.
2. Install the required dependencies by running `pip install mcp requests asyncio shlex` in your terminal.
3. Run the `main.py` script using Python.
4. Follow the on-screen prompts to interact with the connected servers and process queries.

## Conclusion

The provided scripts, `mcp_client_v2.py` and `main.py`, demonstrate how to create a multi-server MCP client that supports connecting to multiple servers, processing queries, and dispatching tool calls based on the call type (tool or resource). The client allows for interoperability with various MCP servers and enables models to communicate effectively with tools and resources in an extensible manner.

These scripts serve as a foundation for building more complex applications that leverage the power of multiple servers using the Model Context Protocol (MCP). By following the documentation and usage instructions provided, you can extend and customize these scripts to meet your specific requirements and integrate them into larger systems seamlessly.

## Thanks
Thanks to @ggozad : https://github.com/ggozad/oterm
