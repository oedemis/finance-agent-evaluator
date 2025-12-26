"""
Finance Agent Evaluator Server.

Main entry point for the Green Agent.
Based on green-agent-template pattern.

Runs two servers:
- A2A server (port 9009): Task/conversation management
- MCP server (port 9020): Tool exposure for purple agents
"""
import argparse
import asyncio
import logging
import os
import threading

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_evaluator")

# Phoenix observability (optional)
_phoenix_observability = None


async def health_check(request):
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({
        "status": "healthy",
        "service": "finance-evaluator",
        "version": "2.0.0"
    })


def create_agent_card(url: str) -> AgentCard:
    """Create the agent card for the finance evaluator."""
    skill = AgentSkill(
        id="finance_research_evaluation",
        name="Financial Research Benchmark",
        description="Evaluates agents on 537 real-world financial research tasks across 9 categories",
        tags=["benchmark", "finance", "evaluation", "gymnasium"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"num_tasks": 10, "categories": ["all"]}}'
        ],
    )
    return AgentCard(
        name="FinanceEvaluator",
        description="Finance Agent Benchmark 2.0 - Multi-dimensional evaluation for financial research agents",
        url=url,
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main():
    global _phoenix_observability

    parser = argparse.ArgumentParser(description="Run the Finance Agent Evaluator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="A2A server port")
    parser.add_argument("--mcp-port", type=int, default=9020, help="MCP server port (0 to disable MCP)")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--data-path", type=str, default="data/public.csv", help="Path to dataset")
    parser.add_argument("--trace-dir", type=str, default="traces", help="Directory to save execution traces")
    parser.add_argument("--phoenix", action="store_true", help="Enable Phoenix observability")
    parser.add_argument("--phoenix-endpoint", type=str, default="http://localhost:6006",
                        help="Phoenix endpoint URL")
    parser.add_argument("--no-llm-judges", action="store_true",
                        help="Disable LLM judges (use heuristic evaluation)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                        help="Model to use for LLM judges")
    parser.add_argument("--debug-a2a", action="store_true",
                        help="Log all A2A protocol messages to trace files")
    args = parser.parse_args()

    # Initialize Phoenix if requested
    if args.phoenix:
        try:
            from observability import setup_observability
            _phoenix_observability = setup_observability(
                project_name="finance-agent-benchmark",
                endpoint=args.phoenix_endpoint,
            )
            if _phoenix_observability:
                logger.info(f"Phoenix observability enabled. View at {args.phoenix_endpoint}")
        except Exception as e:
            logger.warning(f"Could not enable Phoenix: {e}")

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"
    agent_card = create_agent_card(agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(
            data_path=args.data_path,
            trace_dir=args.trace_dir,
            use_llm_judges=not args.no_llm_judges,
            judge_model=args.judge_model,
            debug_a2a=args.debug_a2a,
        ),
        task_store=InMemoryTaskStore(),
    )

    a2a_server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the A2A app
    a2a_app = a2a_server.build()

    # Create wrapper app with health check
    routes = [
        Route("/health", health_check, methods=["GET"]),
    ]

    app = Starlette(routes=routes)
    app.mount("/", a2a_app)

    # Start MCP server in background thread (if enabled)
    mcp_thread = None
    if args.mcp_port > 0:
        try:
            from mcp_server import run_mcp_server_sync

            def start_mcp():
                try:
                    run_mcp_server_sync(host=args.host, port=args.mcp_port)
                except Exception as e:
                    logger.error(f"MCP server failed: {e}")

            mcp_thread = threading.Thread(target=start_mcp, daemon=True, name="MCP-Server")
            mcp_thread.start()
            logger.info(f"🔧 MCP server started on http://{args.host}:{args.mcp_port}")
            logger.info(f"   Tools: edgar_search, google_web_search, parse_html_page, retrieve_information, submit_answer")
        except Exception as e:
            logger.warning(f"Could not start MCP server: {e}")
    else:
        logger.info("MCP server disabled (--mcp-port 0)")

    logger.info(f"Starting Finance Evaluator (A2A) on http://{args.host}:{args.port}")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")
    logger.info(f"Traces will be saved to: {args.trace_dir}/")
    logger.info(f"LLM Judges: {'enabled' if not args.no_llm_judges else 'disabled'}")
    if not args.no_llm_judges:
        logger.info(f"Judge model: {args.judge_model}")
    if args.debug_a2a:
        logger.info("🔍 A2A protocol debugging enabled - all messages will be logged")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
