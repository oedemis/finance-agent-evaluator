"""
Finance Agent Evaluator Server.

Main entry point for the Green Agent.
Based on green-agent-template pattern.
"""
import argparse
import logging

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
    parser = argparse.ArgumentParser(description="Run the Finance Agent Evaluator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--data-path", type=str, default="data/public.csv", help="Path to dataset")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"
    agent_card = create_agent_card(agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(data_path=args.data_path),
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

    logger.info(f"Starting Finance Evaluator on {args.host}:{args.port}")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
