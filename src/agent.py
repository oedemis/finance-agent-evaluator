"""
Finance Evaluator Agent - Main evaluation logic.

Based on green-agent-template pattern.
"""
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import gymnasium as gym
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from dataset import DatasetLoader, Task, RubricItem
from environment import FinancialResearchEnv, register_finance_env
from tools import get_tool_definitions
from tracer import TraceLogger
from prompts import format_prompt
from mcp_server import register_environment, unregister_environment

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

    def __init__(
        self,
        data_path: str = "data/public.csv",
        trace_dir: Optional[str] = None,
        use_llm_judges: bool = True,
        judge_model: str = "gpt-4o-mini",
        debug_a2a: bool = False,
    ):
        self.messenger = Messenger(debug_mode=debug_a2a)
        self.dataset = DatasetLoader(data_path)
        self.tracer = TraceLogger(output_dir=trace_dir)
        self.use_llm_judges = use_llm_judges
        self.judge_model = judge_model
        self.debug_a2a = debug_a2a

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
        custom_query = request.config.get("custom_query")

        # Get the purple agent URL
        agent_url = str(request.participants["agent"])

        # Load tasks - either from dataset or custom query
        if custom_query:
            # Create a custom task for the user's question
            logger.info(f"Running custom query: {custom_query[:100]}...")
            tasks = [Task(
                id="custom_query",
                question=custom_query,
                expert_answer="[Custom query - no expert answer available]",
                category="Custom Query",
                expert_time_mins=0.0,
                rubrics=[],  # No rubrics for custom queries
            )]
        else:
            tasks = self.dataset.get_tasks(categories=categories, limit=num_tasks)

        logger.info(f"Running {len(tasks)} tasks")

        # Start tracing
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        self.tracer.start_run(run_id, dict(request.config), agent_url)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} financial research tasks")
        )

        results: dict[str, Any] = {"tasks": {}}
        category_results: dict[str, list[float]] = {}

        try:
            for i, task in enumerate(tasks):
                logger.info(f"🚀 Running task {i+1}/{len(tasks)}: {task.id}...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"🚀 Running task {i+1}/{len(tasks)}: {task.category}")
                )

                # Start task tracing
                self.tracer.start_task(
                    task_id=task.id,
                    question=task.question,
                    category=task.category,
                    expert_answer=task.expert_answer,
                )

                # Set trace for A2A message logging
                if self.debug_a2a:
                    self.messenger.set_trace(self.tracer.current_task)

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

                    # Emoji status based on pass/fail
                    task_status = "✅" if task_result.get("passed", False) else "❌"
                    benchmark_passed = task_result.get("benchmark_mode", {}).get("passed", False)
                    rl_passed = task_result.get("rl_mode", {}).get("passed", False)
                    benchmark_emoji = "✅" if benchmark_passed else "❌"
                    rl_emoji = "✅" if rl_passed else "❌"

                    logger.info(f"{task_status} Task {task.id} completed | "
                               f"Benchmark: {benchmark_emoji} {task_result.get('benchmark_mode', {}).get('reward', 0):.2f} | "
                               f"RL: {rl_emoji} {task_result['reward']:.2f} | "
                               f"💰 ${task_result.get('costs', {}).get('total_usd', 0):.4f}")

                    # Complete task trace
                    self.tracer.complete_task(task_result)

                except Exception as e:
                    logger.error(f"❌ Task {task.id} failed: {e}")
                    error_result = {
                        "reward": 0.0,
                        "error": str(e),
                        "passed": False,
                    }
                    results["tasks"][task.id] = error_result
                    if task.category not in category_results:
                        category_results[task.category] = []
                    category_results[task.category].append(0.0)

                    # Complete task trace with error
                    self.tracer.complete_task(error_result)

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

            summary = f"""🏆 Finance Agent Benchmark Results
================================
📊 Tasks: {len(tasks)}
✅ Naive Accuracy: {naive_accuracy*100:.1f}%
🎯 Class-Balanced Accuracy: {class_balanced_accuracy*100:.1f}%
📈 Average Reward: {avg_reward:.3f}
⏱️  Time: {time_used:.1f}s

📁 Per-Category Results:
{category_str}"""

            # Complete run trace
            run_trace = self.tracer.complete_run(result_data)
            if run_trace and self.tracer.output_dir:
                trace_file = self.tracer.output_dir / f"{run_trace.run_id}.json"
                summary += f"\n\nTrace saved to: {trace_file}"
                result_data["trace_file"] = str(trace_file)

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
            use_llm_judges=self.use_llm_judges,
            judge_model=self.judge_model,
        )

        # Generate context_id upfront and register environment BEFORE sending message
        # This prevents race condition where purple agent starts calling MCP tools
        # before green agent has a chance to register the environment
        from uuid import uuid4
        context_id = str(uuid4())

        await register_environment(context_id, env)
        logger.info(f"Registered environment for task {task.id} with context_id: {context_id}")

        # Store context_id in messenger so it uses this for the conversation
        self.messenger._context_ids[agent_url] = context_id

        try:
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

                # Log outgoing message
                self.tracer.log_message("user", next_message)

                # Send message to purple agent
                # Pass new_conversation=False even on first message since we already set context_id
                response, artifacts = await self.messenger.talk_to_agent(
                    message=next_message,
                    url=agent_url,
                    new_conversation=is_first_message,
                )

                is_first_message = False

                logger.debug(f"Purple agent response: {response[:200]}...")

                # Extract answer data from A2A artifacts (MCP mode - no text parsing)
                # The purple agent returns structured data as DataPart when submit_answer is called
                from a2a.types import DataPart as A2ADataPart

                # Check artifacts for structured answer data
                answer_submitted = False
                if artifacts:
                    for artifact in artifacts:
                        for part in artifact.parts:
                            if isinstance(part.root, A2ADataPart):
                                # Found structured data - this is the final answer
                                answer_data = part.root.data
                                if isinstance(answer_data, dict) and "answer" in answer_data:
                                    answer_submitted = True
                                    logger.info(f"Purple agent submitted final answer")
                                    break
                        if answer_submitted:
                            break

                # With MCP proxy pattern, purple agent calls tools directly via MCP
                # Each MCP tool call routes to env.step() automatically
                # So we don't need to call env.step() here - just check if the task is done

                if answer_submitted:
                    # Purple agent called submit_answer via MCP (which already called env.step())
                    # Retrieve the final state from the environment (no double-step)
                    # Access unwrapped environment to get our custom attributes
                    unwrapped_env = env.unwrapped
                    terminated = unwrapped_env._last_terminated
                    truncated = unwrapped_env._last_truncated
                    logger.info(f"Task completed: terminated={terminated}, truncated={truncated}")
                    break

                # Check if environment was truncated (max steps reached via MCP calls)
                if env.unwrapped._last_truncated:
                    logger.warning(f"Task truncated: max steps reached")
                    truncated = True
                    break

                # Purple agent is still working - this shouldn't happen in MCP mode
                # because purple agent loops internally and only returns when done
                logger.warning("Purple agent returned without submitting answer (unexpected in MCP mode)")
                next_message = "Please continue your research or submit your final answer."

            # Get final evaluation from environment's last step result
            # Access unwrapped environment to get our custom attributes
            info = env.unwrapped._last_info
            final_result = info.get("evaluation", {})
            task_time = time.time() - start_time

            # Extract judge costs
            judge_costs = final_result.get("costs", {})
            total_cost = info.get("cost", 0.0) + judge_costs.get("total_usd", 0.0)

            return {
                # Legacy format (for backward compatibility)
                "reward": final_result.get("reward", 0.0),
                "factual_accuracy": final_result.get("factual_accuracy", 0.0),
                "has_contradiction": final_result.get("has_contradiction", False),
                "process_quality": final_result.get("process_quality", 0.0),
                "passed": final_result.get("passed", False),

                # Full dual-mode evaluation results
                "benchmark_mode": final_result.get("benchmark_mode", {}),
                "rl_mode": final_result.get("rl_mode", {}),

                # Metadata
                "metadata": {
                    "steps_taken": info.get("step", 0),
                    "time_seconds": task_time,
                    "agent_answer": info.get("agent_answer", ""),
                },

                # Cost breakdown
                "costs": {
                    "total_usd": total_cost,
                    "agent_and_tools": info.get("cost", 0.0),
                    "judges": judge_costs.get("total_usd", 0.0),
                    "judges_breakdown": judge_costs.get("judges_breakdown", {}),
                },
            }

        finally:
            # Cleanup: unregister environment from MCP server (only if it was registered)
            if context_id:
                await unregister_environment(context_id)
                logger.info(f"Unregistered environment for task {task.id} (context: {context_id})")

    def _build_task_prompt(self, task: Task, info: dict) -> str:
        """Build the initial task prompt for the purple agent."""
        tools_json = json.dumps(get_tool_definitions(), indent=2)

        return format_prompt(
            "task_prompt",
            question=task.question,
            tools_json=tools_json,
        )

