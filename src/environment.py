"""
Gymnasium Environment for Financial Research.

This environment simulates a financial research task where an agent
must gather information and provide an answer.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces

from dataset import Task
from tools import ToolExecutor, get_tool_definitions


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
    ):
        super().__init__()

        self.task = task
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Initialize tool executor
        self.tool_executor = ToolExecutor()

        # State
        self.state = EnvironmentState()
        self.agent_answer: Optional[str] = None
        self.trajectory: list[dict] = []

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

        # Execute the tool
        try:
            result, cost = self.tool_executor.execute(
                tool_name=tool_name,
                arguments=arguments,
                data_storage=self.state.data_storage,
            )
            self.state.cost_spent += cost

            # Track documents accessed
            if tool_name == "parse_html_page" and "url" in arguments:
                self.state.documents_accessed.append(arguments["url"])

            observation = self._build_observation(
                obs_type="tool_result",
                content=result,
                metadata={
                    "tool": tool_name,
                    "cost": cost,
                    "step": self.state.current_step,
                }
            )

        except Exception as e:
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
            obs["next_step"] = "Continue research or submit your final answer using submit_answer tool."
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

        This is a simplified evaluation - the full evaluation with
        LLM judges will be added in Phase 3.
        """
        if not self.agent_answer:
            return {
                "reward": 0.0,
                "factual_accuracy": 0.0,
                "has_contradiction": False,
                "process_quality": 0.0,
                "passed": False,
            }

        # Simplified evaluation: check for keyword overlap
        # (This will be replaced with LLM judges in Phase 3)
        answer_lower = self.agent_answer.lower()
        expert_lower = self.task.expert_answer.lower()

        # Count matching words (very basic)
        answer_words = set(answer_lower.split())
        expert_words = set(expert_lower.split())
        overlap = len(answer_words & expert_words)
        max_words = max(len(expert_words), 1)

        # Very rough factual accuracy estimate
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
            "has_contradiction": False,  # Will be evaluated by LLM in Phase 3
            "process_quality": process_quality,
            "passed": reward >= 0.8,
            "rubric_results": [],  # Will be filled by LLM judges
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
