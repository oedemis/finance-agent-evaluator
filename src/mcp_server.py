"""
MCP Server for Finance Agent Evaluator.

Exposes financial research tools via Model Context Protocol (MCP)
to enable purple agents to access tools without text parsing.

Architecture:
- FastMCP server on separate port (default: 8001)
- Tools: edgar_search, google_web_search, parse_html_page, retrieve_information, submit_answer
- State: context_id → data_storage mapping for ephemeral storage
"""
import asyncio
import logging
import time
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.dependencies import Depends

from tools import ToolExecutor, TOOL_COSTS

logger = logging.getLogger("finance_evaluator.mcp")


# ===== State Management =====

class StorageManager:
    """Manages ephemeral data_storage for each context_id."""

    def __init__(self):
        self._storage: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()

    async def get_storage(self, context_id: str) -> dict[str, str]:
        """Get or create storage for a context."""
        async with self._lock:
            if context_id not in self._storage:
                self._storage[context_id] = {}
                logger.info(f"Created storage for context: {context_id}")
            return self._storage[context_id]

    async def cleanup_context(self, context_id: str) -> None:
        """Clean up storage for a completed context."""
        async with self._lock:
            if context_id in self._storage:
                del self._storage[context_id]
                logger.info(f"Cleaned up storage for context: {context_id}")

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return {
            "active_contexts": len(self._storage),
            "total_keys": sum(len(storage) for storage in self._storage.values())
        }


class TelemetryManager:
    """Tracks tool execution telemetry per context for Gymnasium environment."""

    def __init__(self):
        self._telemetry: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()

    async def record_tool_call(
        self,
        context_id: str,
        tool_name: str,
        arguments: dict,
        result: dict,
        cost: float,
        duration_ms: float,
    ) -> None:
        """Record a tool call for later retrieval by the environment."""
        async with self._lock:
            if context_id not in self._telemetry:
                self._telemetry[context_id] = []
                logger.debug(f"Created telemetry tracking for context: {context_id}")

            self._telemetry[context_id].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "cost": cost,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            })

    async def get_telemetry(self, context_id: str) -> list[dict]:
        """Get all telemetry for a context."""
        async with self._lock:
            return self._telemetry.get(context_id, [])

    async def cleanup_context(self, context_id: str) -> None:
        """Clean up telemetry for a completed context."""
        async with self._lock:
            if context_id in self._telemetry:
                del self._telemetry[context_id]
                logger.debug(f"Cleaned up telemetry for context: {context_id}")


# ===== Global State =====

# Storage manager instance (shared across all requests)
_storage_manager = StorageManager()

# Environment registry: context_id → Gymnasium environment instance
# MCP server routes tool calls to the environment (execution layer)
_environments: dict[str, Any] = {}  # Any = FinancialResearchEnv (avoid circular import)
_environments_lock = asyncio.Lock()

# Tool executor instance (legacy - used by old code paths)
_tool_executor = ToolExecutor()


# ===== Environment Registry (MCP as Protocol Layer) =====

async def register_environment(context_id: str, env: Any) -> None:
    """
    Register a Gymnasium environment for a context.

    MCP server routes tool calls to this environment (execution layer).
    Called by agent.py when starting a task.
    """
    async with _environments_lock:
        _environments[context_id] = env
        logger.info(f"Registered environment for context: {context_id}")


async def unregister_environment(context_id: str) -> None:
    """
    Unregister environment after task completion.

    Called by agent.py in finally block.
    """
    async with _environments_lock:
        if context_id in _environments:
            del _environments[context_id]
            logger.info(f"Unregistered environment for context: {context_id}")

        # Also cleanup storage
        await _storage_manager.cleanup_context(context_id)


def get_environment(context_id: str) -> Any:
    """Get environment for context (used by MCP tools)."""
    env = _environments.get(context_id)
    if not env:
        raise RuntimeError(f"No environment registered for context: {context_id}")
    return env


# ===== Dependency Injection =====

async def get_storage(context_id: str) -> dict[str, str]:
    """Dependency: Get data_storage for the current context."""
    return await _storage_manager.get_storage(context_id)


# ===== FastMCP Server =====

mcp = FastMCP(
    name="FinanceEvaluatorTools",
    version="1.0.0",
    instructions="Financial research tools for purple agents: SEC EDGAR search, web search, HTML parsing, RAG retrieval, and answer submission.",
    stateless_http=True,  # Fix for FastMCP Issue #823: Server crashes on client timeout
)


# ===== Tool Definitions =====

@mcp.tool()
async def edgar_search(
    context_id: str,
    query: str,
    form_types: Optional[list[str]] = None,
    ciks: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    top_n_results: int = 10,
) -> dict:
    """
    Search SEC EDGAR database for company filings.

    Args:
        context_id: A2A context identifier for state isolation
        query: Search keywords (e.g., "Apple revenue")
        form_types: Filing types to search (e.g., ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"])
        ciks: Company identifiers (CIK numbers)
        start_date: Start date for filings (format: yyyy-mm-dd)
        end_date: End date for filings (format: yyyy-mm-dd)
        page: Page number for pagination
        top_n_results: Number of results to return

    Returns:
        dict: {"success": bool, "result": str|dict} with search results
    """
    logger.info(f"[{context_id}] MCP edgar_search → routing to environment")

    # Build arguments dict
    arguments = {
        "query": query,
        "top_n_results": top_n_results,
        "page": page,
    }
    if form_types:
        arguments["form_types"] = form_types
    if ciks:
        arguments["ciks"] = ciks
    if start_date:
        arguments["start_date"] = start_date
    if end_date:
        arguments["end_date"] = end_date

    # MCP as protocol layer: Route to environment (execution layer)
    env = get_environment(context_id)
    observation, reward, terminated, truncated, info = env.step({
        "name": "edgar_search",
        "arguments": arguments
    })

    logger.info(f"[{context_id}] edgar_search completed: cost=${info.get('cost', 0):.4f}")

    # Return observation as MCP result
    return {"success": True, "result": observation}


@mcp.tool()
async def google_web_search(
    context_id: str,
    search_query: str,
) -> dict:
    """
    Search the web using Google (via SerpAPI).

    Args:
        context_id: A2A context identifier for state isolation
        search_query: Search query string

    Returns:
        dict: {"success": bool, "result": str|dict} with search results
    """
    logger.info(f"[{context_id}] MCP google_web_search → routing to environment")

    arguments = {"search_query": search_query}

    # MCP as protocol layer: Route to environment (execution layer)
    env = get_environment(context_id)
    observation, reward, terminated, truncated, info = env.step({
        "name": "google_web_search",
        "arguments": arguments
    })

    logger.info(f"[{context_id}] google_web_search completed: cost=${info.get('cost', 0):.4f}")

    # Return observation as MCP result
    return {"success": True, "result": observation}


@mcp.tool()
async def parse_html_page(
    context_id: str,
    url: str,
    key: str,
) -> dict:
    """
    Parse HTML content from a URL and store it for later retrieval.

    Args:
        context_id: A2A context identifier for state isolation
        url: URL of the page to parse
        key: Storage key for later retrieval (use in retrieve_information)

    Returns:
        dict: {"success": bool, "result": str} with parsed content preview
    """
    logger.info(f"[{context_id}] MCP parse_html_page → routing to environment")

    arguments = {"url": url, "key": key}

    # MCP as protocol layer: Route to environment (execution layer)
    env = get_environment(context_id)
    observation, reward, terminated, truncated, info = env.step({
        "name": "parse_html_page",
        "arguments": arguments
    })

    logger.info(f"[{context_id}] parse_html_page completed: cost=${info.get('cost', 0):.4f}")

    # Return observation as MCP result
    return {"success": True, "result": observation}


@mcp.tool()
async def retrieve_information(
    context_id: str,
    prompt: str,
    input_character_ranges: Optional[dict[str, list[int]]] = None,
) -> dict:
    """
    Use LLM to extract information from stored documents (via RAG).

    Args:
        context_id: A2A context identifier for state isolation
        prompt: Prompt with placeholders like "Extract revenue from {{earnings_report}}"
        input_character_ranges: Optional character slicing per key (e.g., {"earnings_report": [0, 5000]})

    Returns:
        dict: {"success": bool, "result": str, "usage": dict} with LLM analysis and token usage
    """
    logger.info(f"[{context_id}] MCP retrieve_information → routing to environment")

    arguments = {"prompt": prompt}
    if input_character_ranges:
        arguments["input_character_ranges"] = input_character_ranges

    # MCP as protocol layer: Route to environment (execution layer)
    env = get_environment(context_id)
    observation, reward, terminated, truncated, info = env.step({
        "name": "retrieve_information",
        "arguments": arguments
    })

    logger.info(f"[{context_id}] retrieve_information completed: cost=${info.get('cost', 0):.4f}")

    # Return observation as MCP result
    result = {"success": True, "result": observation}

    # DEBUG: Log the actual payload being returned
    logger.info(f"[{context_id}] retrieve_information PAYLOAD: type={type(observation)}, len={len(str(observation)) if observation else 0}, preview={str(observation)[:200] if observation else 'None'}...")

    return result


@mcp.tool()
async def submit_answer(
    context_id: str,
    answer: str,
    sources: list[dict[str, str]],
) -> dict:
    """
    Submit final answer to complete the research task.

    Args:
        context_id: A2A context identifier for state isolation
        answer: The complete answer with all key facts, numbers, and details
        sources: List of sources [{"url": "...", "name": "..."}]

    Returns:
        dict: {"success": True, "result": "Answer submitted"} - this triggers task completion
    """
    logger.info(f"[{context_id}] MCP submit_answer → routing to environment")

    # Validate sources format (pre-validation in protocol layer)
    if not isinstance(sources, list):
        return {
            "success": False,
            "result": "sources must be a list of dicts with 'url' and 'name' keys"
        }

    for source in sources:
        if not isinstance(source, dict) or "url" not in source or "name" not in source:
            return {
                "success": False,
                "result": "Each source must be a dict with 'url' and 'name' keys"
            }

    arguments = {"answer": answer, "sources": sources}

    # MCP as protocol layer: Route to environment (execution layer)
    env = get_environment(context_id)
    observation, reward, terminated, truncated, info = env.step({
        "name": "submit_answer",
        "arguments": arguments
    })

    logger.info(f"[{context_id}] submit_answer completed: reward={reward}, terminated={terminated}")

    # Return minimal response to avoid FastMCP Client timeout on large payloads
    # Purple agent doesn't need the data - it already has the answer and will send via A2A
    # The evaluation result is retrieved by green agent from environment directly
    return {
        "success": True,
        "message": "Answer submitted and evaluated successfully"
    }


@mcp.tool()
async def get_storage_stats() -> dict:
    """
    Get statistics about storage usage (admin/debug tool).

    Returns:
        dict: {"active_contexts": int, "total_keys": int}
    """
    stats = _storage_manager.get_stats()
    logger.info(f"Storage stats: {stats}")
    return stats


# ===== Server Lifecycle =====

def run_mcp_server_sync(host: str = "127.0.0.1", port: int = 8001):
    """
    Run the MCP server (synchronous).

    Args:
        host: Host to bind to
        port: Port to bind to
    """
    logger.info(f"Starting MCP server on {host}:{port}")
    logger.info(f"Exposing tools: edgar_search, google_web_search, parse_html_page, retrieve_information, submit_answer")

    try:
        # FastMCP's run() is synchronous
        # Use "http" transport (recommended for new projects) or "sse" for backward compatibility
        mcp.run(transport="http", host=host, port=port)
    except Exception as e:
        logger.error(f"MCP server failed: {e}")
        raise


# ===== Standalone Execution =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Finance Evaluator MCP Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info(f"Finance Evaluator MCP Server v1.0.0")
    logger.info(f"Starting on http://{args.host}:{args.port}")

    run_mcp_server_sync(host=args.host, port=args.port)
