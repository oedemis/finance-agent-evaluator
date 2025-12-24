"""
Finance Evaluator Agent - Main evaluation logic.

Based on green-agent-template pattern.
"""
import json
import logging
import time
from typing import Any

import gymnasium as gym
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from dataset import DatasetLoader, Task
from environment import FinancialResearchEnv, register_finance_env
from tools import get_tool_definitions

logger = logging.getLogger("finance_evaluator")

# Register the Gymnasium environment
register_finance_env()


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class FinanceEvaluatorAgent:
    """Green agent that evaluates purple agents on financial research tasks."""

    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []  # All config keys have defaults

    def __init__(self, data_path: str = "data/public.csv"):
        self.messenger = Messenger()
        self.dataset = DatasetLoader(data_path)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the evaluation request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run the evaluation.

        Args:
            message: The incoming evaluation request
            updater: Report progress and results
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await self._run_evaluation(request, updater)

    async def _run_evaluation(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """Execute the evaluation logic."""
        logger.info(f"Starting finance evaluation: {request}")
        start_time = time.time()

        # Get configuration with defaults
        num_tasks = request.config.get("num_tasks", 10)
        categories = request.config.get("categories", ["all"])
        max_steps = request.config.get("max_steps", 50)
        timeout = request.config.get("timeout", 600)

        # Get the purple agent URL
        agent_url = str(request.participants["agent"])

        # Load tasks
        tasks = self.dataset.get_tasks(categories=categories, limit=num_tasks)
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

            # Class-balanced accuracy
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
            self.messenger.reset()

    async def _run_single_task(
        self,
        agent_url: str,
        task: Task,
        max_steps: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Run a single financial research task."""

        env: FinancialResearchEnv = gym.make(
            "FinancialResearch-v0",
            task=task,
            max_steps=max_steps,
        )

        terminated = False
        truncated = False
        observation, info = env.reset()

        # Build the initial task prompt
        task_prompt = self._build_task_prompt(task, info)

        next_message = task_prompt
        is_first_message = True
        start_time = time.time()

        while not terminated and not truncated:
            if time.time() - start_time > timeout:
                truncated = True
                break

            logger.debug(f"Sending to purple agent: {next_message[:200]}...")

            # Send message to purple agent
            response = await self.messenger.talk_to_agent(
                message=next_message,
                url=agent_url,
                new_conversation=is_first_message,
            )
            is_first_message = False

            logger.debug(f"Purple agent response: {response[:200]}...")

            # Parse the action
            try:
                action = self._parse_agent_response(response)
            except Exception as e:
                logger.warning(f"Failed to parse agent response: {e}")
                action = {"name": "_error", "arguments": {"message": str(e)}}

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            logger.debug(f"Environment step: reward={reward}, terminated={terminated}")

            if terminated or truncated:
                break

            next_message = observation

        # Get final evaluation
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

        # Try <json>...</json> tags
        match = re.search(r'<json>\s*(.*?)\s*</json>', response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try markdown code blocks
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
            return json.loads(response)
