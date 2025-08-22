#!/usr/bin/env python3
"""
CLI client pour un LLM OpenAI-compatible + plusieurs serveurs MCP locaux.

LM Studio: gpt-oss-20B
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
API_BASE = "http://192.168.10.54:1234/v1"
DEBUG    = True
APPROVE  = "Strict"  # False, True, "Strict" -- Strict is only for sudo commands

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

def _assistant_message_dict_from_choice(choice) -> dict:
    """Reconstitue un message assistant (dict) à partir de la réponse OpenAI."""
    msg = {"role": "assistant", "content": choice.content or ""}
    if getattr(choice, "tool_calls", None):
        msg["tool_calls"] = []
        for tc in choice.tool_calls:
            msg["tool_calls"].append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            })
    return msg

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
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=openai_tools,
                temperature=TEMPERATURE,
                seed=SEED,
                stream=False,
            )

            elapsed_ns = time.perf_counter_ns() - start_ns
            usage = completion.usage
            print(
                f"⏱️ Temps de traitement : {nanosecondes_vers_lisible(elapsed_ns)}\n"
                f"📊 Tokens : prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, total={usage.total_tokens}"
            )

            choice = completion.choices[0].message

            reasoning = getattr(choice, "reasoning", None)
            if reasoning:
                print("\n🧠 Raisonnement du modèle :")
                print(reasoning)

            if not getattr(choice, "tool_calls", None):
                final = choice.content or ""
                print("\n=== Réponse du modèle ===")
                #print(final.strip())
                console.print(Markdown(final.strip())) ## markdown display
                print("==========================")
                print("\n🗨️ (Ctrl-D pour quitter) :")
                messages.append({"role": "assistant", "content": final})
                break

            assistant_msg = _assistant_message_dict_from_choice(choice)
            messages.append(assistant_msg)

            for tc in choice.tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    try:
                        args = json.loads(raw_args.replace("'", '"'))
                    except Exception:
                        args = {}

                if DEBUG:
                    print(f"  DEBUG: 🔧 Appel MCP → {tool_name}({args})")

                session = tool_sessions.get(tool_name)
                if not session:
                    tool_output = f"Outil inconnu: {tool_name}"
                else:
                    sub_name = tool_name.split(".", 1)[-1]
                    tool_output = await call_mcp_tool(session, sub_name, args)

                if DEBUG:
                    print("  DEBUG: 📥 Résultat MCP :", (tool_output[:500] + "…") if len(tool_output) > 500 else tool_output)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })

# ------------------------------- Main ------------------------------
async def run():
    sessions = []
    openai_tools = []
    tool_sessions = {}

    async with AsyncExitStack() as stack:
        # Ouvre et garde TOUTES les connexions MCP pendant tout le REPL
        for idx, params in enumerate(MCP_SERVERS):
            #prefix = f"srv{idx}"
            prefix = os.path.splitext(os.path.basename(params.args[0]))[0]

            # 1) ouvrir le transport stdio
            read, write = await stack.enter_async_context(stdio_client(params))

            # 2) ouvrir la session protocole MCP
            session = ClientSession(read, write)
            await stack.enter_async_context(session)

            # 3) initialiser la session
            await session.initialize()
            sessions.append(session)

            # 4) récupérer les tools de CE serveur
            tools_result = await session.list_tools()
            mcp_tools = getattr(tools_result, "tools", tools_result)

            print(f" 🔧 Outils MCP ({prefix}):\n", [t.name for t in mcp_tools])

            # 5) exposer ces tools au LLM avec un préfixe pour le routage
            openai_tools.extend(mcp_tools_to_openai(mcp_tools, prefix))
            for t in mcp_tools:
                tool_sessions[f"{prefix}.{t.name}"] = session

        # 👉 Toutes les connexions sont maintenant ouvertes et stables.
        # On peut lancer le REPL ; l’ExitStack fermera proprement à la sortie.
        await interactive_loop(sessions, openai_tools, tool_sessions)


def main():
    asyncio.run(run())

if __name__ == "__main__":
    main()
