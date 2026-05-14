from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from typing import List, Dict, Any, Optional

import httpx
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


@tool
def context7_search(libraryName: str, query: str) -> Dict[str, Any]:
    """
    Search Context7 library documentation via HTTP.
    Reads configuration from environment variables that are typically
    set via config.yaml:
      - CONTEXT7_API_KEY  (required)
      - CONTEXT7_BASE_URL (optional, defaults to https://context7.com)
    """
    api_key = os.getenv("CONTEXT7_API_KEY") or os.getenv("CONTEXT7_TOKEN")
    if not api_key:
        return {"error": "CONTEXT7_API_KEY is not set in environment"}

    base_url = os.getenv("CONTEXT7_BASE_URL", "https://context7.com").rstrip("/")
    url = f"{base_url}/api/v2/libs/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"libraryName": libraryName, "query": query}
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Context7 HTTP request failed: {e}"}

class MCPManager:
    def __init__(self, mcp_config: Any):
        self.config = mcp_config
        self.exit_stack = contextlib.AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tools: List[Any] = []
        self._loop = asyncio.new_event_loop()
        self._thread: Optional[threading.Thread] = None

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self):
        if not self.config or not getattr(self.config, 'enabled', False):
            return
        
        servers = getattr(self.config, 'servers', [])
        if not servers:
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Connect synchronously by blocking until connected
        future = asyncio.run_coroutine_threadsafe(self._connect_all(servers), self._loop)
        future.result()

    async def _connect_all(self, servers: List[Dict[str, Any]]):
        for server in servers:
            name = server.get("name") or ""
            cmd = server.get("command")
            if not cmd:
                # If no command but this is the Context7 server, register HTTP fallback tool
                if name.lower() == "context7":
                    self.tools.append(context7_search)
                continue

            args = server.get("args", [])
            env = server.get("env", None)

            params = StdioServerParameters(command=cmd, args=args, env=env)

            try:
                read, write = await self.exit_stack.enter_async_context(stdio_client(params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.sessions.append(session)

                mcp_tools = await load_mcp_tools(session)
                self.tools.extend(mcp_tools)
            except FileNotFoundError:
                # If the binary is missing but the server is 'context7', fall back to HTTP tool
                if name.lower() == "context7":
                    self.tools.append(context7_search)
                # Do not crash pipeline
                continue
            except Exception:
                # Any other MCP startup error should not crash the main pipeline
                continue

    def stop(self):
        if self._thread and self._thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(self.exit_stack.aclose(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)

    def get_tools(self) -> List[Any]:
        return self.tools
