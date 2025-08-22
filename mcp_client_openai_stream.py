#!/usr/bin/env python3
"""
CLI client pour un LLM OpenAI-compatible + plusieurs serveurs MCP locaux.
LM Studio:
    - Context Length : 32768
    - Evaluation batch size : 8192

Usage:
    $ python client_mcp.py
    > Peux-tu lister le contenu de ~/scripts ?
"""

import os
import sys
import time
import json
import asyncio
import logging
from contextlib import AsyncExitStack
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

# ----------------------------- Imports -----------------------------
try:
    from openai import OpenAI
except ImportError:
    print("❌ Vous devez installer le package 'openai' (pip install openai).")
    sys.exit(1)

try:
    # mcp >= 1.2 style
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ Vous devez installer le package 'mcp' (pip install mcp).")
    sys.exit(1)

# ----------------------------- Logging ---------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-46s \t (%(filename)s:%(lineno)d)",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.ERROR)

# Rich console
console = Console()

# -------------------------- Configuration --------------------------
API_KEY  = os.getenv("OPENAI_API_KEY", "lmstudio")
if API_KEY == "lmstudio":
    logger.warning('⚠️  No OpenAI API Key found, using default.')
API_BASE = "http://192.168.10.123:1234/v1"
DEBUG    = True
APPROVE  = "Strict"  # False, True, "Strict"

MODEL_NAME  = "openai/gpt-oss-20b"
TEMPERATURE = 0.0
SEED        = 1234567890

client = OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=None)

# Liste de serveurs MCP à lancer
MCP_SERVERS = [
    StdioServerParameters(command="python3", args=["mcp_server_filesystem.py"]),
    StdioServerParameters(command="python3", args=["mcp_server_pgsql.py"]),
    StdioServerParameters(command="python3", args=["mcp_server_shell.py"]),
]

# ------------------------------ Utils ------------------------------
def nanosecondes_vers_lisible(ns: int) -> str:
    secondes = ns / 1_000_000_000
    minutes  = int(secondes // 60)
    rest_sec = secondes % 60
    ms       = (rest_sec - int(rest_sec)) * 10

    parts = []
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if rest_sec >= 1:
        parts.append(f"{int(rest_sec)} seconde{'s' if int(rest_sec) > 1 else ''}")
    if rest_sec < 1 and ms > 0:
        parts.append(f"{int(ms)} dixième{'s' if int(ms) > 1 else ''}")
    return ", ".join(parts) or "moins d'un dixième de seconde"

def mcp_tools_to_openai(tools: list, prefix: str) -> list:
    """
    Convertit les MCP Tools (types.Tool) en format OpenAI ChatCompletions.
    On préfixe les noms d'outils pour éviter les collisions.
    """
    out = []
    for t in tools:
        params = getattr(t, "inputSchema", None)
        if params is None and hasattr(t, "input_schema"):
            params = getattr(t, "input_schema")
        if params is None:
            params = {"type": "object", "properties": {}}

        tool_name = f"{prefix}.{getattr(t, 'name', 'tool')}"

        out.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": getattr(t, "description", "") or "",
                "parameters": params,
            },
        })
    return out

async def call_mcp_tool(session: ClientSession, name: str, args: dict) -> str:
    """Appelle un outil MCP et renvoie un texte agrégé.
       Pour le serveur shell, on peut demander approbation à l'utilisateur."""
    
    # Interception shell
    if "shell" in getattr(session, "name", "") or name == "run_command":
        command = args.get("command", "")
        needs_confirm = False

        if APPROVE is True:
            needs_confirm = True
        elif APPROVE == "Strict" and "sudo" in command.split():
            needs_confirm = True

        if needs_confirm:
            print(f"\n⚠️  Demande d'exécution d'une commande shell :\n{command}")
            confirm = input("Voulez-vous exécuter ? [y/N] ").strip().lower()
            if confirm != "y":
                return "⛔ Commande annulée par l'utilisateur."

    try:
        res = await session.call_tool(name=name, arguments=args)
        contents = getattr(res, "content", res)
        chunks = []
        for c in contents:
            if getattr(c, "type", "") == "text" and hasattr(c, "text") and c.text:
                chunks.append(c.text)
        return "\n".join(chunks) if chunks else ""
    except Exception as e:
        return f"Erreur MCP: {e}"

# ------------------------ LLM + MCP Orchestration -------------------
async def interactive_loop(all_sessions, openai_tools, tool_sessions):
    """Boucle REPL avec historique en mémoire et support multi-serveurs MCP."""

    messages = []  # historique volatile

    print("🗨️  Entrez votre question (Ctrl-D pour quitter) :")
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        start_ns = time.perf_counter_ns()

        while True:
            stream = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                tools       = openai_tools,
                temperature = TEMPERATURE,
                seed        = SEED,
                stream      = True,
            )

            in_progress_tool_calls = {}
            full_content = ""
            is_tool_call_response = False
            usage_info = None
            
            console.print("\n[bold cyan]=== Réponse du modèle ===[/bold cyan]")

            # --- MODIFICATION: USE RICH LIVE FOR MARKDOWN STREAMING ---
            with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    
                    # 1. Stream and accumulate content for a text response
                    if delta and delta.content:
                        full_content += delta.content
                        # Re-render the entire accumulated content as Markdown
                        live.update(Markdown(full_content), refresh=True)

                    # 2. Stream and reconstruct tool calls
                    if delta and delta.tool_calls:
                        is_tool_call_response = True
                        for tc_chunk in delta.tool_calls:
                            index = tc_chunk.index
                            if index not in in_progress_tool_calls:
                                in_progress_tool_calls[index] = {
                                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                                }
                            if tc_chunk.id:
                                in_progress_tool_calls[index]["id"] = tc_chunk.id
                            if tc_chunk.function and tc_chunk.function.name:
                                in_progress_tool_calls[index]["function"]["name"] += tc_chunk.function.name
                            if tc_chunk.function and tc_chunk.function.arguments:
                                in_progress_tool_calls[index]["function"]["arguments"] += tc_chunk.function.arguments

                    # 3. Capture usage stats from the final chunk
                    if chunk.usage:
                        usage_info = chunk.usage
            
            # --- END OF STREAM & LIVE DISPLAY ---
            
            # Print usage info if available
            if usage_info:
                elapsed_ns = time.perf_counter_ns() - start_ns
                print(
                    f"\n⏱️ Temps de traitement : {nanosecondes_vers_lisible(elapsed_ns)}\n"
                    f"📊 Tokens : prompt={usage_info.prompt_tokens}, "
                    f"completion={usage_info.completion_tokens}, total={usage_info.total_tokens}"
                )

            console.print("[bold cyan]=========================[/bold cyan]")

            if not is_tool_call_response:
                messages.append({"role": "assistant", "content": full_content})
                print("\n🗨️ (Ctrl-D pour quitter) :")
                break

            # --- Tool Call Execution Logic (remains mostly the same) ---
            completed_tool_calls = [v for k, v in sorted(in_progress_tool_calls.items())]
            
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": completed_tool_calls
            }
            messages.append(assistant_msg)

            for tc in completed_tool_calls:
                tool_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                
                console.print(f"\n[bold magenta]▶️ Exécution de l'outil : {tool_name}({raw_args})[/bold magenta]")

                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    console.print(f"[red]Erreur: Arguments JSON invalides pour {tool_name}.[/red]")
                    args = {}

                session = tool_sessions.get(tool_name)
                if not session:
                    tool_output = f"Outil inconnu: {tool_name}"
                else:
                    sub_name = tool_name.split(".", 1)[-1]
                    tool_output = await call_mcp_tool(session, sub_name, args)

                console.print(f"[green]✅ Résultat de {tool_name} :[/green]")
                console.print(tool_output)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })
            
            console.print("\n[bold blue]... Renvoi des résultats de l'outil au modèle ...[/bold blue]")


# ------------------------------- Main ------------------------------
async def run():
    sessions = []
    openai_tools = []
    tool_sessions = {}

    async with AsyncExitStack() as stack:
        for idx, params in enumerate(MCP_SERVERS):
            prefix = os.path.splitext(os.path.basename(params.args[0]))[0]
            read, write = await stack.enter_async_context(stdio_client(params))
            session = ClientSession(read, write)
            await stack.enter_async_context(session)
            await session.initialize()
            sessions.append(session)
            tools_result = await session.list_tools()
            mcp_tools = getattr(tools_result, "tools", tools_result)
            print(f" 🔧 Outils MCP ({prefix}):\n", [t.name for t in mcp_tools])
            openai_tools.extend(mcp_tools_to_openai(mcp_tools, prefix))
            for t in mcp_tools:
                tool_sessions[f"{prefix}.{t.name}"] = session

        await interactive_loop(sessions, openai_tools, tool_sessions)


def main():
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Au revoir !")
        sys.exit(0)

if __name__ == "__main__":
    main()
