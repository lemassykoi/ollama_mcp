import sys
import asyncio
import logging
from mcp_client_v2 import MultiServerMCPClient
import colorama

BLUE         = colorama.Fore.BLUE
YELLOW       = colorama.Fore.YELLOW
RESET        = colorama.Style.RESET_ALL

# Initialize logger
logging.basicConfig(
    format = '%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)',
    #stream = sys.stdout,
    level  = logging.DEBUG
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

servers_to_connect = {
    "time": "mcp_server_time.py",
    "pgsql": "mcp_server_pgsql.py",
    "domoticz": "mcp_server_domoticz.py",
    "filesystem": "mcp_server_filesystem.py",
}

async def main():
    client = MultiServerMCPClient(servers_to_connect)
    await client.gather_connections()
    messages = None
    while True:
        user_query = input(BLUE + "User => ").strip()
        print(RESET)
        if user_query in ['exit', 'quit', 'cls']:
            break
        messages = await client.process_query(user_query, messages)
        print(YELLOW + messages[-1]['content'] + RESET)

    print(messages)
    print('User Exit')
    await client.cleanup()
    

if __name__ == '__main__':
    asyncio.run(main())

sys.exit(0)
