import requests
import asyncio
import logging
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import BaseModel, Field

logging.basicConfig(
    format='%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Create a server instance
server = Server(
    "DOMOTICZ-MCP-SERVER",
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

class DeviceStatusInput(BaseModel):
    idx: int = Field(..., description="Device IDX")

class ControlDeviceInput(BaseModel):
    idx: int = Field(..., description="Device IDX")
    command: str = Field(..., description="Command to send ('On', 'Off', 'Toggle')")

# -------------------------------
# Domoticz API Functions
# -------------------------------
DOMOTICZ_USERNAME = 'python'
DOMOTICZ_PASSWORD = 'password'
domoticz_ip       = '10.0.0.1'
domoticz_port     = '80'

DOMOTICZ_URL = f'http://{domoticz_ip}:{domoticz_port}'
DOMOTICZ_API_PATH  = "/json.htm"
DOMOTICZ_PLAN_NAME = "LLM"     # A plan with some devices attached (the devices we want to expose to LLM)

@register_tool
async def get_status() -> str:
    """
    Returns the status of Domoticz instance.
    """
    url = f"{DOMOTICZ_URL}{DOMOTICZ_API_PATH}"
    status_params = {
         "type": "command",
         "param": "getversion",
    }
    try:
        status_plans = requests.get(url=url, params=status_params, auth=(DOMOTICZ_USERNAME, DOMOTICZ_PASSWORD)).json()
        return status_plans
    except Exception as e:
        raise ValueError(str(e))

@register_tool
async def list_devices() -> str:
    """
    Returns a list of all usable devices.
    """
    # 1. Retrieve plans to get the idx for the "LLM" plan.
    url = f"{DOMOTICZ_URL}{DOMOTICZ_API_PATH}"
    plan_params = {
         "type": "command",
         "param": "getplans",
         "order": "name",
         "used": "true"
    }
    response_plans = requests.get(url=url, params=plan_params, auth=(DOMOTICZ_USERNAME, DOMOTICZ_PASSWORD)).json()
    json_plans = response_plans.get("result", [])
    # Map plan names to idx values.
    plan_mapping = {plan["Name"]: int(plan["idx"]) for plan in json_plans if "Name" in plan and "idx" in plan}
    if DOMOTICZ_PLAN_NAME not in plan_mapping:
         raise ValueError(f"Plan '{DOMOTICZ_PLAN_NAME}' not found.")
    
    plan_idx = plan_mapping[DOMOTICZ_PLAN_NAME]

    # 2. Retrieve all devices.
    response_devices = await get_devices()
    json_devices = response_devices.get("result", [])

    # 3. Filter functions.
    def filter_device_from_plan(device: dict, plan_idx: int):
         # Ensure device has a PlanIDs field (expected to be a list).
         if "PlanIDs" in device and isinstance(device["PlanIDs"], list):
              if plan_idx in device["PlanIDs"]:
                   return device
         return None

    def filter_device_data(device: dict):
         return {
              "Name": device.get("Name"),
              "idx": device.get("idx"),
              "Data": device.get("Data"),
              "SwitchType": device.get("SwitchType")
         }

    # 4. Filter devices that belong to the specified plan.
    filtered_devices_from_plan = [
         filter_device_from_plan(device, plan_idx) for device in json_devices
         if filter_device_from_plan(device, plan_idx) is not None
    ]
    filtered_devices = [
         filter_device_data(device) for device in filtered_devices_from_plan
         if filter_device_data(device) is not None
    ]

    return filtered_devices

@register_tool
async def get_device_status(data: DeviceStatusInput) -> str:
    """
    Returns the status of a specific device.
    Battery level at 255% indicates that the device is not powered by battery, then you can safely ignore this value.
    """
    devices = await get_devices()
    logger.debug(devices)
    for device in devices.get("result", []):
        if device["idx"] == str(data.idx):
            return device
    logger.error(f"Device {data.idx} not found in the device list.")
    raise ValueError({"error": "Device not found"})

get_device_status.input_model = DeviceStatusInput
get_device_status.input_schema = DeviceStatusInput.model_json_schema()

@register_tool
async def control_device(data: ControlDeviceInput) -> str:
    """
    Controls a specific device (e.g., turn on/off). If you don't know the IDX, call the 'list_devices' tool to get the IDX associated with a device name.
    """
    response = requests.get(f"{DOMOTICZ_URL}/json.htm?type=command&param=switchlight&idx={data.idx}&switchcmd={data.command}", auth=(DOMOTICZ_USERNAME, DOMOTICZ_PASSWORD))
    if response.status_code == 200:
        logger.info(f'Control device {data.idx} => OK')
        return response.json()
    else:
        logger.error(f'Control device {data.idx} => FAILED')
        raise ValueError({"error": "Failed to control device"})

control_device.input_model = ControlDeviceInput
control_device.input_schema = ControlDeviceInput.model_json_schema()


# -------------------------------
# Other Funcs
# -------------------------------
async def get_devices():
    """
    Returns a list of all devices.
    """
    response = requests.get(f"{DOMOTICZ_URL}/json.htm?type=devices&used=true", auth=(DOMOTICZ_USERNAME, DOMOTICZ_PASSWORD))
    if response.status_code == 200:
        logger.info('Get all devices => OK')
        return response.json()
    else:
        logger.error('Get all devices => FAILED')
        return {"error": "Failed to fetch devices list"}


# -------------------------------
# Start MCP Server
# -------------------------------

async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="Domoticz MCP Server",
                server_version="0.1.0",
                instructions="You are in charge of a Home Automation Instance: Domoticz. You have access to some switchable devices, and some informations.",
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
