"""
Finance Agent Evaluator - Green agent that runs Finance Agent Benchmark on purple agents.

This agent:
1. Loads financial research tasks from dataset
2. Sets up Gymnasium environments for each task
3. Sends task prompts to the purple agent (the agent being tested)
4. Parses the purple agent's tool-call responses
5. Steps through the environment and evaluates results
"""
import argparse
import asyncio
import json
import logging
import time
from typing import Any

import gymnasium as gym
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

from dataset import DatasetLoader, Task
from environment import FinancialResearchEnv, register_finance_env
from tools import get_tool_definitions


# Health check handler
async def health_check(request):
    """Health check endpoint for Docker and load balancers."""
    return JSONResponse({"status": "healthy", "service": "finance-evaluator", "version": "2.0.0"})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_evaluator")

SUBMIT_ANSWER_ACTION = "submit_answer"

# Register the Gymnasium environment
register_finance_env()


class FinanceEvaluator(GreenAgent):
    """Green agent that evaluates purple agents on financial research tasks."""

    def __init__(self, data_path: str = "data/public.csv"):
        self._required_roles = ["agent"]
        self._required_config_keys = []  # No required config, all have defaults
        self._tool_provider = ToolProvider()
        self._dataset = DatasetLoader(data_path)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting finance evaluation: {req}")
        start_time = time.time()

        # Get configuration with defaults
        num_tasks = req.config.get("num_tasks", 10)
        categories = req.config.get("categories", ["all"])
        max_steps = req.config.get("max_steps", 50)
        timeout = req.config.get("timeout", 600)

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])

        # Load tasks
        tasks = self._dataset.get_tasks(categories=categories, limit=num_tasks)
        logger.info(f"Running {len(tasks)} tasks")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} financial research tasks")
        )

        results: dict[str, Any] = {"tasks": {}}
        category_results: dict[str, list[float]] = {}

        try:
            for i, task in enumerate(tasks):
                logger.info(f"Running task {i+1}/{len(tasks)}: {task.id}...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {i+1}/{len(tasks)}: {task.category}")
                )

                try:
                    task_result = await self._run_single_task(
                        agent_url=agent_url,
                        task=task,
                        max_steps=max_steps,
                        timeout=timeout,
                    )
                    results["tasks"][task.id] = task_result

                    # Track per-category
                    if task.category not in category_results:
                        category_results[task.category] = []
                    category_results[task.category].append(task_result["reward"])

                    logger.info(f"Task {task.id} completed with reward: {task_result['reward']:.2f}")
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    results["tasks"][task.id] = {
                        "reward": 0.0,
                        "error": str(e),
                        "passed": False,
                    }
                    if task.category not in category_results:
                        category_results[task.category] = []
                    category_results[task.category].append(0.0)

            # Calculate aggregate metrics
            time_used = time.time() - start_time
            all_rewards = [r["reward"] for r in results["tasks"].values()]

            naive_accuracy = sum(1 for r in all_rewards if r >= 0.8) / len(all_rewards) if all_rewards else 0
            avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

            # Class-balanced accuracy (average per category)
            per_category_accuracy = {}
            for cat, rewards in category_results.items():
                per_category_accuracy[cat] = sum(1 for r in rewards if r >= 0.8) / len(rewards) if rewards else 0
            class_balanced_accuracy = sum(per_category_accuracy.values()) / len(per_category_accuracy) if per_category_accuracy else 0

            result_data = {
                "naive_accuracy": naive_accuracy,
                "class_balanced_accuracy": class_balanced_accuracy,
                "avg_reward": avg_reward,
                "total_tasks": len(tasks),
                "passed_tasks": sum(1 for r in all_rewards if r >= 0.8),
                "per_category": per_category_accuracy,
                "task_results": results["tasks"],
                "time_used": time_used,
            }

            # Format summary
            category_str = "\n".join(
                f"  {cat}: {acc*100:.1f}% ({sum(1 for r in category_results[cat] if r >= 0.8)}/{len(category_results[cat])})"
                for cat, acc in per_category_accuracy.items()
            )

            summary = f"""Finance Agent Benchmark Results
================================
Tasks: {len(tasks)}
Naive Accuracy: {naive_accuracy*100:.1f}%
Class-Balanced Accuracy: {class_balanced_accuracy*100:.1f}%
Average Reward: {avg_reward:.3f}
Time: {time_used:.1f}s

Per-Category Results:
{category_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )

        finally:
            self._tool_provider.reset()

    async def _run_single_task(
        self,
        agent_url: str,
        task: Task,
        max_steps: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Run a single financial research task and return the result."""

        env: FinancialResearchEnv = gym.make(
            "FinancialResearch-v0",
            task=task,
            max_steps=max_steps,
        )

        terminated = False
        truncated = False
        observation, info = env.reset()

        # Build the initial task description for the purple agent
        task_prompt = self._build_task_prompt(task, info)

        # Start a new conversation with the purple agent
        next_message = task_prompt
        is_first_message = True
        start_time = time.time()

        while not terminated and not truncated:
            # Check timeout
            if time.time() - start_time > timeout:
                truncated = True
                break

            logger.debug(f"Sending to purple agent: {next_message[:200]}...")

            # Send message to purple agent
            response = await self._tool_provider.talk_to_agent(
                message=next_message,
                url=agent_url,
                new_conversation=is_first_message,
            )
            is_first_message = False

            logger.debug(f"Purple agent response: {response[:200]}...")

            # Parse the purple agent's action
            try:
                action = self._parse_agent_response(response)
            except Exception as e:
                logger.warning(f"Failed to parse agent response: {e}")
                # Send error back and let agent retry
                action = {"name": "_error", "arguments": {"message": str(e)}}

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            logger.debug(f"Environment step: reward={reward}, terminated={terminated}")

            if terminated or truncated:
                break

            next_message = observation

        # Get final evaluation from environment
        final_result = info.get("evaluation", {})

        return {
            "reward": final_result.get("reward", 0.0),
            "factual_accuracy": final_result.get("factual_accuracy", 0.0),
            "has_contradiction": final_result.get("has_contradiction", False),
            "process_quality": final_result.get("process_quality", 0.0),
            "steps_taken": info.get("step", 0),
            "cost": info.get("cost", 0.0),
            "time": time.time() - start_time,
            "passed": final_result.get("reward", 0.0) >= 0.8,
            "agent_answer": info.get("agent_answer", ""),
        }

    def _build_task_prompt(self, task: Task, info: dict) -> str:
        """Build the initial task prompt for the purple agent."""
        tools_json = json.dumps(get_tool_definitions(), indent=2)

        return f"""You are a financial research agent. Your task is to answer the following question by researching financial data.

QUESTION:
{task.question}

AVAILABLE TOOLS:
{tools_json}

INSTRUCTIONS:
1. Use the available tools to research and gather information
2. You can search SEC EDGAR filings, search Google, parse web pages, and retrieve information from stored documents
3. When you have enough information, submit your final answer using the submit_answer tool
4. Be thorough but efficient - avoid redundant tool calls

RESPONSE FORMAT:
Respond with a JSON object wrapped in <json>...</json> tags:
<json>
{{
  "name": "tool_name",
  "arguments": {{...}}
}}
</json>

Example responses:
<json>
{{"name": "edgar_search", "arguments": {{"query": "revenue", "form_types": ["10-K"], "ciks": [], "start_date": "2023-01-01", "end_date": "2024-12-31", "page": "1", "top_n_results": 5}}}}
</json>

<json>
{{"name": "submit_answer", "arguments": {{"answer": "Based on my research..."}}}}
</json>

Begin your research now.
"""

    def _parse_agent_response(self, response: str) -> dict:
        """Parse the purple agent's response to extract the action."""
        import re

        json_str = None

        # Try to extract JSON from <json>...</json> tags
        match = re.search(r'<json>\s*(.*?)\s*</json>', response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to extract JSON from markdown code blocks
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    json_str = match.group(1)

        if json_str:
            return json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            return json.loads(response)


def finance_evaluator_agent_card(name: str, url: str) -> AgentCard:
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
        name=name,
        description="Finance Agent Benchmark 2.0 - Multi-dimensional evaluation for financial research agents",
        url=url,
        version="2.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the Finance Agent Evaluator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument("--data-path", type=str, default="data/public.csv", help="Path to dataset")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    agent = FinanceEvaluator(data_path=args.data_path)
    executor = GreenExecutor(agent)
    agent_card = finance_evaluator_agent_card("FinanceEvaluator", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the A2A app and add health check route
    a2a_app = a2a_server.build()

    # Create wrapper app with health check
    routes = [
        Route("/health", health_check, methods=["GET"]),
    ]

    # Mount A2A app
    app = Starlette(routes=routes)
    app.mount("/", a2a_app)

    uvicorn_config = uvicorn.Config(app, host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)

    logger.info(f"Starting Finance Evaluator on {args.host}:{args.port}")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")

    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
