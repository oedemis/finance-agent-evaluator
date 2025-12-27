"""
Trace logging for Finance Agent Benchmark.

Saves detailed execution traces to JSON files for debugging and analysis.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolCall:
    """A single tool call by the purple agent."""
    step: int
    tool_name: str
    arguments: dict
    result: str
    cost: float
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class A2AProtocolMessage:
    """Raw A2A protocol message for debugging."""
    direction: str  # "sent" or "received"
    message_type: str  # "request", "response", "event", etc.
    payload: dict  # Full JSON-RPC message
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentMessage:
    """A message from the purple agent."""
    step: int
    role: str  # "user" (green->purple) or "assistant" (purple->green)
    content: str
    parsed_action: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TaskTrace:
    """Complete trace of a single task execution."""
    task_id: str
    question: str
    category: str
    expert_answer: str

    # Execution trace
    messages: list[AgentMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    a2a_protocol_messages: list[A2AProtocolMessage] = field(default_factory=list)

    # Final results
    agent_answer: str = ""
    reward: float = 0.0
    factual_accuracy: float = 0.0
    process_quality: float = 0.0
    passed: bool = False

    # Full evaluation (includes benchmark_mode, rl_mode, costs, etc.)
    full_evaluation: Optional[dict] = None

    # Metadata
    steps_taken: int = 0
    total_cost: float = 0.0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def add_message(self, role: str, content: str, parsed_action: Optional[dict] = None):
        """Add a message to the trace."""
        self.messages.append(AgentMessage(
            step=len(self.messages),
            role=role,
            content=content,
            parsed_action=parsed_action,
        ))

    def add_tool_call(self, tool_name: str, arguments: dict, result: str, cost: float, duration_ms: float):
        """Add a tool call to the trace."""
        self.tool_calls.append(ToolCall(
            step=len(self.tool_calls),
            tool_name=tool_name,
            arguments=arguments,
            result=result[:10000],  # Truncate long results
            cost=cost,
            duration_ms=duration_ms,
        ))

    def add_a2a_message(self, direction: str, message_type: str, payload: dict):
        """Add an A2A protocol message to the trace for debugging."""
        self.a2a_protocol_messages.append(A2AProtocolMessage(
            direction=direction,
            message_type=message_type,
            payload=payload,
        ))

    def complete(self, result: dict):
        """Mark task as complete with results."""
        self.completed_at = datetime.now().isoformat()

        # Extract metadata
        metadata = result.get("metadata", {})
        self.agent_answer = metadata.get("agent_answer", result.get("agent_answer", ""))
        self.steps_taken = metadata.get("steps_taken", result.get("steps_taken", 0))
        self.duration_seconds = metadata.get("time_seconds", result.get("time", 0.0))

        # Extract costs
        costs = result.get("costs", {})
        self.total_cost = costs.get("total_usd", result.get("cost", 0.0))

        # Legacy fields for backward compatibility
        self.reward = result.get("reward", 0.0)
        self.factual_accuracy = result.get("factual_accuracy", 0.0)
        self.process_quality = result.get("process_quality", 0.0)
        self.passed = result.get("passed", False)
        self.error = result.get("error")

        # Store full evaluation (includes benchmark_mode, rl_mode, costs, etc.)
        self.full_evaluation = {
            "benchmark_mode": result.get("benchmark_mode", {}),
            "rl_mode": result.get("rl_mode", {}),
            "costs": result.get("costs", {}),
            "metadata": result.get("metadata", {}),
            # Legacy fields
            "reward": self.reward,
            "factual_accuracy": self.factual_accuracy,
            "process_quality": self.process_quality,
            "passed": self.passed,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Use full_evaluation if available, otherwise legacy format
        evaluation = self.full_evaluation if self.full_evaluation else {
            "reward": self.reward,
            "factual_accuracy": self.factual_accuracy,
            "process_quality": self.process_quality,
            "passed": self.passed,
        }

        result = {
            "task_id": self.task_id,
            "question": self.question,
            "category": self.category,
            "expert_answer": self.expert_answer,
            "agent_answer": self.agent_answer,
            "messages": [asdict(m) for m in self.messages],
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "evaluation": evaluation,  # Full evaluation with benchmark_mode, rl_mode, costs
            "metrics": {
                "steps_taken": self.steps_taken,
                "total_cost": self.total_cost,
                "duration_seconds": self.duration_seconds,
            },
            "timestamps": {
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "error": self.error,
        }

        # Include A2A protocol messages if available (for debugging)
        if self.a2a_protocol_messages:
            result["a2a_protocol_messages"] = [asdict(m) for m in self.a2a_protocol_messages]

        return result


@dataclass
class RunTrace:
    """Complete trace of an evaluation run."""
    run_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Configuration
    config: dict = field(default_factory=dict)
    purple_agent_url: str = ""

    # Task traces
    tasks: list[TaskTrace] = field(default_factory=list)

    # Aggregate results
    total_tasks: int = 0
    passed_tasks: int = 0
    naive_accuracy: float = 0.0
    class_balanced_accuracy: float = 0.0
    avg_reward: float = 0.0
    total_time: float = 0.0
    total_cost: float = 0.0

    def add_task(self, task_trace: TaskTrace):
        """Add a task trace."""
        self.tasks.append(task_trace)

    def complete(self, results: dict):
        """Mark run as complete with aggregate results."""
        self.completed_at = datetime.now().isoformat()
        self.total_tasks = results.get("total_tasks", len(self.tasks))
        self.passed_tasks = results.get("passed_tasks", 0)
        self.naive_accuracy = results.get("naive_accuracy", 0.0)
        self.class_balanced_accuracy = results.get("class_balanced_accuracy", 0.0)
        self.avg_reward = results.get("avg_reward", 0.0)
        self.total_time = results.get("time_used", 0.0)
        self.total_cost = sum(t.total_cost for t in self.tasks)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "config": self.config,
            "purple_agent_url": self.purple_agent_url,
            "tasks": [t.to_dict() for t in self.tasks],
            "aggregate": {
                "total_tasks": self.total_tasks,
                "passed_tasks": self.passed_tasks,
                "naive_accuracy": self.naive_accuracy,
                "class_balanced_accuracy": self.class_balanced_accuracy,
                "avg_reward": self.avg_reward,
                "total_time": self.total_time,
                "total_cost": self.total_cost,
            },
            "timestamps": {
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
        }

    def to_leaderboard_dict(self) -> dict:
        """Convert to leaderboard-optimized format for $2M competition display.

        Structure optimized for ranking and comparison:
        - Top-level: Primary ranking metrics (benchmark mode)
        - Second-level: Component scores (factual, contradiction, process)
        - Third-level: Efficiency metrics (cost, time)
        - Bottom: Full trace data for verification
        """
        # Calculate aggregate scores across all tasks
        all_evals = [t.full_evaluation for t in self.tasks if t.full_evaluation]

        if not all_evals:
            # No evaluations available
            avg_benchmark_reward = 0.0
            avg_factual = 0.0
            avg_contradiction = 0.0
            avg_process = 0.0
            avg_rl_reward = 0.0
            benchmark_pass_rate = 0.0
            rl_pass_rate = 0.0
        else:
            # Extract benchmark mode metrics
            benchmark_rewards = [e.get("benchmark_mode", {}).get("reward", 0.0) for e in all_evals]
            avg_benchmark_reward = sum(benchmark_rewards) / len(benchmark_rewards)
            benchmark_pass_rate = sum(1 for e in all_evals
                                     if e.get("benchmark_mode", {}).get("passed", False)) / len(all_evals)

            # Extract component scores
            avg_factual = sum(e.get("factual_accuracy", 0.0) for e in all_evals) / len(all_evals)
            avg_contradiction = sum(e.get("benchmark_mode", {}).get("contradiction", {}).get("score", 1.0)
                                   for e in all_evals) / len(all_evals)
            avg_process = sum(e.get("process_quality", 0.0) for e in all_evals) / len(all_evals)

            # Extract RL mode metrics
            rl_rewards = [e.get("rl_mode", {}).get("reward", 0.0) for e in all_evals]
            avg_rl_reward = sum(rl_rewards) / len(rl_rewards)
            rl_pass_rate = sum(1 for e in all_evals
                              if e.get("rl_mode", {}).get("passed", False)) / len(all_evals)

        # Calculate per-category metrics
        per_category_metrics = {}
        category_tasks = {}
        for task in self.tasks:
            cat = task.category
            if cat not in category_tasks:
                category_tasks[cat] = []
            category_tasks[cat].append(task)

        for cat, tasks in category_tasks.items():
            cat_evals = [t.full_evaluation for t in tasks if t.full_evaluation]
            if cat_evals:
                per_category_metrics[cat] = {
                    "benchmark_reward": sum(e.get("benchmark_mode", {}).get("reward", 0.0)
                                          for e in cat_evals) / len(cat_evals),
                    "pass_rate": sum(1 for e in cat_evals
                                   if e.get("benchmark_mode", {}).get("passed", False)) / len(cat_evals),
                    "task_count": len(tasks),
                }

        return {
            # Submission identification
            "submission": {
                "run_id": self.run_id,
                "agent_url": self.purple_agent_url,
                "timestamp": self.completed_at or self.started_at,
                "config": self.config,
            },

            # PRIMARY METRICS (for ranking and leaderboard display)
            "leaderboard_metrics": {
                "class_balanced_accuracy": self.class_balanced_accuracy,  # ⭐ MAIN RANKING METRIC
                "naive_accuracy": self.naive_accuracy,
                "benchmark_reward": avg_benchmark_reward,  # 0.7*factual + 0.3*contradiction
                "benchmark_pass_rate": benchmark_pass_rate,  # % tasks passed benchmark criteria
            },

            # COMPONENT SCORES (understanding what drives performance)
            "component_scores": {
                "factual_accuracy": avg_factual,      # % of rubrics matched
                "contradiction_score": avg_contradiction,  # 1.0 = no contradictions
                "process_quality": avg_process,       # ⭐ Separate process metric
            },

            # EFFICIENCY METRICS (cost-performance tradeoff)
            "efficiency": {
                "avg_cost_per_task": self.total_cost / self.total_tasks if self.total_tasks > 0 else 0.0,
                "avg_time_per_task": self.total_time / self.total_tasks if self.total_tasks > 0 else 0.0,
                "total_cost": self.total_cost,
                "total_time": self.total_time,
                "cost_accuracy_ratio": (self.total_cost / self.class_balanced_accuracy
                                       if self.class_balanced_accuracy > 0 else float('inf')),
            },

            # RL MODE METRICS (for training and improvement)
            "rl_metrics": {
                "rl_reward": avg_rl_reward,  # 0.5*factual + 0.2*contra + 0.3*process
                "rl_pass_rate": rl_pass_rate,
            },

            # PER-CATEGORY BREAKDOWN
            "per_category": per_category_metrics,

            # DETAILED RESULTS (for verification and debugging)
            "detailed_results": {
                "total_tasks": self.total_tasks,
                "passed_tasks": self.passed_tasks,
                "task_summaries": [
                    {
                        "task_id": t.task_id,
                        "category": t.category,
                        "benchmark_reward": t.full_evaluation.get("benchmark_mode", {}).get("reward", 0.0) if t.full_evaluation else 0.0,
                        "passed": t.full_evaluation.get("benchmark_mode", {}).get("passed", False) if t.full_evaluation else False,
                        "cost": t.total_cost,
                        "time": t.duration_seconds,
                    }
                    for t in self.tasks
                ],
            },

            # Link to full trace
            "trace_file": f"{self.run_id}.json",
        }


class TraceLogger:
    """Logger that saves execution traces to JSON files."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the trace logger.

        Args:
            output_dir: Directory to save traces. If None, traces are not saved to disk.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.current_run: Optional[RunTrace] = None
        self.current_task: Optional[TaskTrace] = None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, run_id: str, config: dict, purple_agent_url: str) -> RunTrace:
        """Start a new evaluation run."""
        self.current_run = RunTrace(
            run_id=run_id,
            config=config,
            purple_agent_url=purple_agent_url,
        )
        return self.current_run

    def start_task(self, task_id: str, question: str, category: str, expert_answer: str) -> TaskTrace:
        """Start a new task trace."""
        self.current_task = TaskTrace(
            task_id=task_id,
            question=question,
            category=category,
            expert_answer=expert_answer,
        )
        return self.current_task

    def log_message(self, role: str, content: str, parsed_action: Optional[dict] = None):
        """Log a message exchange."""
        if self.current_task:
            self.current_task.add_message(role, content, parsed_action)

    def log_tool_call(self, tool_name: str, arguments: dict, result: str, cost: float, duration_ms: float):
        """Log a tool call."""
        if self.current_task:
            self.current_task.add_tool_call(tool_name, arguments, result, cost, duration_ms)

    def complete_task(self, result: dict):
        """Complete the current task and add to run."""
        if self.current_task:
            self.current_task.complete(result)
            if self.current_run:
                self.current_run.add_task(self.current_task)
            self.current_task = None

    def complete_run(self, results: dict):
        """Complete the run and save to disk."""
        if self.current_run:
            self.current_run.complete(results)
            self._save_run()
            return self.current_run
        return None

    def _save_run(self):
        """Save the current run to disk in multiple formats."""
        if not self.output_dir or not self.current_run:
            return

        # Save full trace (for debugging)
        trace_file = self.output_dir / f"{self.current_run.run_id}.json"
        with open(trace_file, "w") as f:
            json.dump(self.current_run.to_dict(), f, indent=2)

        # Save leaderboard format (for ranking/comparison)
        leaderboard_file = self.output_dir / f"{self.current_run.run_id}_leaderboard.json"
        with open(leaderboard_file, "w") as f:
            json.dump(self.current_run.to_leaderboard_dict(), f, indent=2)

        # Save summary
        summary_file = self.output_dir / f"{self.current_run.run_id}_summary.json"
        summary = {
            "run_id": self.current_run.run_id,
            "config": self.current_run.config,
            "aggregate": {
                "total_tasks": self.current_run.total_tasks,
                "passed_tasks": self.current_run.passed_tasks,
                "naive_accuracy": self.current_run.naive_accuracy,
                "class_balanced_accuracy": self.current_run.class_balanced_accuracy,
                "avg_reward": self.current_run.avg_reward,
                "total_time": self.current_run.total_time,
                "total_cost": self.current_run.total_cost,
            },
            "per_task": {
                t.task_id: {
                    "category": t.category,
                    "reward": t.reward,
                    "passed": t.passed,
                    "steps": t.steps_taken,
                    "cost": t.total_cost,
                    "time": t.duration_seconds,
                }
                for t in self.current_run.tasks
            },
            "timestamps": {
                "started_at": self.current_run.started_at,
                "completed_at": self.current_run.completed_at,
            },
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
