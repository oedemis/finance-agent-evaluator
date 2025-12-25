"""
Gymnasium Environment for Financial Research.

This environment simulates a financial research task where an agent
must gather information and provide an answer.
Includes OpenTelemetry tracing for Phoenix observability.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
from gymnasium import spaces

from dataset import Task
from tools import ToolExecutor, get_tool_definitions, TOOL_COSTS

logger = logging.getLogger("finance_evaluator.environment")

# Optional OpenTelemetry tracing with OpenInference semantic conventions
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from openinference.semconv.trace import SpanAttributes, ToolCallAttributes
    _tracer = trace.get_tracer("finance_evaluator.environment")
    _HAS_OTEL = True

    # OpenInference semantic conventions for Phoenix display
    # See: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
    INPUT_VALUE = SpanAttributes.INPUT_VALUE
    OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
    OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
except ImportError:
    _HAS_OTEL = False
    _tracer = None
    INPUT_VALUE = OUTPUT_VALUE = OPENINFERENCE_SPAN_KIND = None
    SpanAttributes = None
    ToolCallAttributes = None


@dataclass
class EnvironmentState:
    """State tracked during an episode."""
    # Research Progress
    documents_accessed: list[str] = field(default_factory=list)
    facts_extracted: dict[str, Any] = field(default_factory=dict)
    data_storage: dict[str, str] = field(default_factory=dict)

    # Resource Tracking
    tools_used: list[str] = field(default_factory=list)
    tool_call_count: dict[str, int] = field(default_factory=dict)
    cost_spent: float = 0.0
    time_elapsed: float = 0.0

    # Episode Metadata
    current_step: int = 0
    start_time: float = 0.0


class FinancialResearchEnv(gym.Env):
    """
    Gymnasium environment for financial research tasks.

    The agent must use tools to research financial information
    and provide an answer to a question.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task: Task,
        max_steps: int = 50,
        render_mode: Optional[str] = None,
        use_llm_judges: bool = True,
        judge_model: str = "gpt-4o-mini",
    ):
        super().__init__()

        self.task = task
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_llm_judges = use_llm_judges
        self.judge_model = judge_model

        # Initialize tool executor
        self.tool_executor = ToolExecutor()

        # LLM Judges (lazy loaded)
        self._evaluation_orchestrator = None

        # State
        self.state = EnvironmentState()
        self.agent_answer: Optional[str] = None
        self.trajectory: list[dict] = []
        self.tool_calls: list[dict] = []  # Track tool calls for process evaluation

        # Define action and observation spaces
        # Action space: JSON string representing tool call
        self.action_space = spaces.Text(max_length=10000)
        # Observation space: JSON string representing observation
        self.observation_space = spaces.Text(max_length=100000)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset state
        self.state = EnvironmentState()
        self.state.start_time = time.time()
        self.agent_answer = None
        self.trajectory = []
        self.tool_calls = []

        # Build initial observation
        observation = self._build_observation(
            obs_type="initial",
            content=f"Task: {self.task.question}",
            metadata={"category": self.task.category}
        )

        info = {
            "task_id": self.task.id,
            "category": self.task.category,
            "tools": get_tool_definitions(),
        }

        return observation, info

    def step(self, action: dict) -> tuple[str, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Dictionary with 'name' and 'arguments' keys

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.state.current_step += 1
        self.state.time_elapsed = time.time() - self.state.start_time

        # Parse action if it's a string
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return self._handle_error("Invalid JSON in action"), 0.0, False, False, {}

        tool_name = action.get("name", "")
        arguments = action.get("arguments", {})

        # Track tool usage
        self.state.tools_used.append(tool_name)
        self.state.tool_call_count[tool_name] = self.state.tool_call_count.get(tool_name, 0) + 1

        # Check for submit_answer
        if tool_name == "submit_answer":
            self.agent_answer = arguments.get("answer", "")
            observation = self._build_observation(
                obs_type="final",
                content="Answer submitted.",
                metadata={"answer": self.agent_answer}
            )

            # Evaluate the answer
            evaluation = self._evaluate_answer()
            reward = evaluation["reward"]

            info = {
                "task_id": self.task.id,
                "step": self.state.current_step,
                "cost": self.state.cost_spent,
                "agent_answer": self.agent_answer,
                "evaluation": evaluation,
            }

            # Log trajectory
            self.trajectory.append({
                "step": self.state.current_step,
                "action": action,
                "reward": reward,
                "terminated": True,
            })

            return observation, reward, True, False, info

        # Handle error action
        if tool_name == "_error":
            observation = self._handle_error(arguments.get("message", "Unknown error"))
            return observation, 0.0, False, False, {"step": self.state.current_step}

        # Execute the tool with optional OpenTelemetry tracing
        def _execute_tool():
            return self.tool_executor.execute(
                tool_name=tool_name,
                arguments=arguments,
                data_storage=self.state.data_storage,
            )

        try:
            if _HAS_OTEL and _tracer and SpanAttributes:
                # Use OpenInference TOOL span kind for proper Phoenix display
                # See: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md#tool-calls
                with _tracer.start_as_current_span(
                    f"Tool: {tool_name}",
                    kind=SpanKind.INTERNAL,
                    attributes={
                        OPENINFERENCE_SPAN_KIND: "TOOL",  # Mark as TOOL for Phoenix
                        SpanAttributes.TOOL_NAME: tool_name,
                        SpanAttributes.TOOL_DESCRIPTION: f"Finance research tool: {tool_name}",
                        SpanAttributes.TOOL_PARAMETERS: json.dumps(arguments),
                        # Input/Output for Phoenix UI
                        INPUT_VALUE: json.dumps(arguments),
                        # Custom attributes
                        "tool.step": self.state.current_step,
                        "tool.cost": TOOL_COSTS.get(tool_name, 0.0),
                    }
                ) as span:
                    result_dict, cost = _execute_tool()

                    # Set output and status
                    result_str = str(result_dict.get("result", ""))
                    success = result_dict.get("success", False)

                    span.set_attribute(OUTPUT_VALUE, result_str[:2000] if result_str else "")
                    span.set_attribute("tool.success", success)
                    span.set_attribute("tool.actual_cost", cost)

                    if success:
                        span.set_status(Status(StatusCode.OK))
                    else:
                        span.set_status(Status(StatusCode.ERROR, result_str[:100]))
            else:
                result_dict, cost = _execute_tool()

            # Extract result string from {"success": bool, "result": ...} format
            success = result_dict.get("success", False)
            result_content = result_dict.get("result", "")

            # Handle retrieve_information usage metadata and ACTUAL cost
            actual_cost = cost  # Default to fixed cost
            if tool_name == "retrieve_information" and "usage" in result_dict:
                usage = result_dict["usage"]
                logger.info(f"retrieve_information usage: {usage}")
                # Use actual cost if available
                if "cost_usd" in usage:
                    actual_cost = usage["cost_usd"]
                    logger.info(f"Using actual cost ${actual_cost:.6f} instead of fixed ${cost:.6f}")

            self.state.cost_spent += actual_cost

            # Track tool calls for process evaluation
            self.tool_calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": str(result_content)[:2000],  # Truncate for judges
                "success": success,
                "cost": actual_cost,  # Use actual cost, not fixed
                "step": self.state.current_step,
            })

            # Track documents accessed
            if tool_name == "parse_html_page" and "url" in arguments:
                self.state.documents_accessed.append(arguments["url"])

            observation = self._build_observation(
                obs_type="tool_result",
                content=str(result_content),
                metadata={
                    "tool": tool_name,
                    "cost": actual_cost,  # Use actual cost
                    "success": success,
                    "step": self.state.current_step,
                }
            )

        except Exception as e:
            if _HAS_OTEL and _tracer:
                span = trace.get_current_span()
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
            observation = self._handle_error(f"Tool execution failed: {str(e)}")

        # Calculate intermediate reward (small penalties/bonuses)
        intermediate_reward = self._calculate_intermediate_reward(tool_name, arguments)

        # Check for truncation
        truncated = self.state.current_step >= self.max_steps

        info = {
            "task_id": self.task.id,
            "step": self.state.current_step,
            "cost": self.state.cost_spent,
            "time": self.state.time_elapsed,
        }

        if truncated:
            # Evaluate even if truncated
            self.agent_answer = ""
            evaluation = self._evaluate_answer()
            info["evaluation"] = evaluation

        # Log trajectory
        self.trajectory.append({
            "step": self.state.current_step,
            "action": action,
            "observation_preview": observation[:200],
            "reward": intermediate_reward,
        })

        return observation, intermediate_reward, False, truncated, info

    def _build_observation(
        self,
        obs_type: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Build an observation string for the agent."""
        obs = {
            "type": obs_type,
            "content": content,
            "metadata": metadata or {},
            "state_summary": {
                "step": self.state.current_step,
                "cost_so_far": self.state.cost_spent,
                "time_so_far": self.state.time_elapsed,
                "documents_accessed": len(self.state.documents_accessed),
                "data_keys": list(self.state.data_storage.keys()),
            }
        }

        # Add instruction based on type
        if obs_type == "tool_result":
            obs["next_step"] = "Continue research or submit your final answer using FINAL ANSWER: format."
        elif obs_type == "error":
            obs["next_step"] = "Please fix the issue and try again."

        return json.dumps(obs, indent=2)

    def _handle_error(self, message: str) -> str:
        """Handle an error during execution."""
        return self._build_observation(
            obs_type="error",
            content=message,
            metadata={"error": True}
        )

    def _calculate_intermediate_reward(self, tool_name: str, arguments: dict) -> float:
        """Calculate small intermediate reward/penalty."""
        reward = 0.0

        # Redundancy penalty - same tool with same arguments
        action_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        if action_key in [f"{t}:{json.dumps(a, sort_keys=True)}" for t, a in zip(
            self.state.tools_used[:-1],
            [self.trajectory[i]["action"].get("arguments", {}) for i in range(len(self.trajectory))]
            if self.trajectory else []
        )]:
            reward -= 0.05  # Redundancy penalty

        # Small cost penalty
        reward -= 0.001 * self.state.cost_spent

        return reward

    def _evaluate_answer(self) -> dict:
        """
        Evaluate the agent's answer against the ground truth.

        Uses LLM judges if enabled, otherwise falls back to heuristic evaluation.
        """
        if not self.agent_answer:
            return {
                "reward": 0.0,
                "factual_accuracy": 0.0,
                "has_contradiction": False,
                "process_quality": 0.0,
                "passed": False,
            }

        if self.use_llm_judges:
            return self._evaluate_with_llm_judges()
        else:
            return self._evaluate_heuristic()

    def _evaluate_with_llm_judges(self) -> dict:
        """Evaluate using LLM judges (async wrapper)."""
        try:
            # Lazy load the orchestrator to avoid import issues
            if self._evaluation_orchestrator is None:
                from judges import EvaluationOrchestrator
                self._evaluation_orchestrator = EvaluationOrchestrator(model=self.judge_model)

            # Build trajectory string for judges
            trajectory = self._build_trajectory_string()

            # Convert rubrics to list of dicts
            rubrics = []
            if self.task.rubrics:
                rubrics = [{"criteria": r.criteria, "operator": r.operator} for r in self.task.rubrics]

            # Run async evaluation in sync context
            async def run_evaluation():
                return await self._evaluation_orchestrator.evaluate(
                    question=self.task.question,
                    expert_answer=self.task.expert_answer,
                    agent_answer=self.agent_answer,
                    trajectory=trajectory,
                    tool_calls=self.tool_calls,
                    rubrics=rubrics,
                )

            # Execute async code in sync context
            try:
                loop = asyncio.get_running_loop()
                # Already in async context, use ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_evaluation())
                    result = future.result(timeout=120)  # 2 minute timeout
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                result = asyncio.run(run_evaluation())

            # Add passed flag
            result["passed"] = result.get("reward", 0.0) >= 0.8

            logger.info(f"LLM Judge evaluation: reward={result['reward']:.3f}, "
                       f"factual={result['factual_accuracy']:.3f}, "
                       f"process={result['process_quality']:.3f}")

            return result

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}, falling back to heuristic")
            return self._evaluate_heuristic()

    def _build_trajectory_string(self) -> str:
        """Build a readable trajectory string for judges."""
        lines = []
        for tc in self.tool_calls:
            lines.append(f"Step {tc['step']}: {tc['tool_name']}")
            if tc.get('arguments'):
                args_str = json.dumps(tc['arguments'], indent=2)
                lines.append(f"  Arguments: {args_str}")
            if tc.get('result'):
                result_preview = tc['result'][:500]
                lines.append(f"  Result: {result_preview}...")
            lines.append("")
        return "\n".join(lines)

    def _evaluate_heuristic(self) -> dict:
        """Fallback heuristic evaluation (no LLM calls)."""
        answer_lower = self.agent_answer.lower()
        expert_lower = self.task.expert_answer.lower()

        # Count matching words (basic overlap)
        answer_words = set(answer_lower.split())
        expert_words = set(expert_lower.split())
        overlap = len(answer_words & expert_words)
        max_words = max(len(expert_words), 1)

        # Rough factual accuracy estimate
        factual_accuracy = min(1.0, overlap / max_words * 2)

        # Process quality based on tool usage
        unique_tools = len(set(self.state.tools_used))
        total_tools = len(self.state.tools_used)
        redundancy = 1.0 - (self.state.tool_call_count.get(
            max(self.state.tool_call_count, key=self.state.tool_call_count.get, default=""),
            1
        ) - 1) * 0.1 if self.state.tool_call_count else 1.0
        process_quality = min(1.0, max(0.0, redundancy * (unique_tools / max(total_tools, 1) + 0.5)))

        # Combined reward
        reward = 0.7 * factual_accuracy + 0.3 * process_quality

        return {
            "reward": reward,
            "factual_accuracy": factual_accuracy,
            "has_contradiction": False,
            "process_quality": process_quality,
            "passed": reward >= 0.8,
            "rubric_results": [],
        }

    def render(self) -> None:
        """Render the environment state."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.state.current_step} ===")
            print(f"Task: {self.task.question[:100]}...")
            print(f"Tools used: {self.state.tools_used}")
            print(f"Cost: ${self.state.cost_spent:.4f}")
            print(f"Time: {self.state.time_elapsed:.1f}s")

    def close(self) -> None:
        """Clean up resources."""
        pass


def register_finance_env():
    """Register the finance environment with Gymnasium."""
    gym.register(
        id="FinancialResearch-v0",
        entry_point="environment:FinancialResearchEnv",
    )
