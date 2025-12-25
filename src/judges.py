"""
LLM-based evaluation judges for Finance Agent Benchmark.

Uses LiteLLM for flexible model selection and Phoenix for observability.
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import litellm
from litellm import completion_cost

from prompts import load_prompt, format_prompt

logger = logging.getLogger("finance_evaluator.judges")


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation."""
    score: float
    details: dict[str, Any]
    raw_response: str
    cost_usd: float = 0.0  # Actual cost from LiteLLM


class BaseJudge(ABC):
    """Base class for LLM-based judges."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def get_prompt_name(self) -> str:
        """Return the name of the prompt template to use."""
        pass

    @abstractmethod
    def get_prompt_kwargs(self, **kwargs) -> dict:
        """Prepare kwargs for the prompt template."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> JudgeResult:
        """Parse the LLM response into a JudgeResult."""
        pass

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from various response formats."""
        # Try markdown code blocks first
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Try plain code blocks
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object directly
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: try parsing the whole response
        return json.loads(response)

    async def evaluate(self, **kwargs) -> JudgeResult:
        """Run the judge evaluation."""
        prompt_kwargs = self.get_prompt_kwargs(**kwargs)
        prompt = format_prompt(self.get_prompt_name(), **prompt_kwargs)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Calculate actual cost
            actual_cost = 0.0
            try:
                actual_cost = completion_cost(completion_response=response)
                logger.info(f"{self.__class__.__name__} cost: ${actual_cost:.6f} (model: {self.model})")
            except Exception as cost_err:
                logger.warning(f"Failed to calculate judge cost: {cost_err}")

            response_text = response.choices[0].message.content
            result = self.parse_response(response_text)
            result.cost_usd = actual_cost
            return result

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return JudgeResult(
                score=0.0,
                details={"error": str(e)},
                raw_response="",
                cost_usd=0.0,
            )


class FactualAccuracyJudge(BaseJudge):
    """Judge for evaluating factual accuracy of agent answers.

    Uses correctness rubrics to check if key facts are present (RECALL).
    """

    def get_prompt_name(self) -> str:
        return "factual_accuracy_judge"

    def get_prompt_kwargs(
        self,
        question: str,
        expert_answer: str,
        agent_answer: str,
        correctness_rubrics: list[str] = None,
        **kwargs,
    ) -> dict:
        rubrics_str = ""
        if correctness_rubrics:
            rubrics_str = "\n".join(
                f"{i+1}. {rubric}"
                for i, rubric in enumerate(correctness_rubrics)
            )
        else:
            rubrics_str = "(No specific rubric items provided)"

        return {
            "question": question,
            "expert_answer": expert_answer,
            "agent_answer": agent_answer,
            "rubrics": rubrics_str,
        }

    def parse_response(self, response: str) -> JudgeResult:
        try:
            data = self._extract_json(response)
            return JudgeResult(
                score=float(data.get("score", 0.0)),
                details={
                    "rubric_scores": data.get("rubric_scores", {}),
                    "reasoning": data.get("reasoning", ""),
                    "key_facts_found": data.get("key_facts_found", []),
                    "key_facts_missing": data.get("key_facts_missing", []),
                    "errors": data.get("errors", []),
                },
                raw_response=response,
            )
        except Exception as e:
            logger.warning(f"Failed to parse factual accuracy response: {e}")
            return JudgeResult(
                score=0.5,  # Default to middle score on parse failure
                details={"parse_error": str(e)},
                raw_response=response,
            )


class ContradictionJudge(BaseJudge):
    """Judge for detecting contradictions in agent responses.

    Checks if agent's answer contains information that CONTRADICTS the expected answer.
    This check is LENIENT - it allows different wording/structure but flags wrong facts.
    Helps reduce FALSE NEGATIVES (correct answers marked as wrong).
    """

    def get_prompt_name(self) -> str:
        return "contradiction_judge"

    def get_prompt_kwargs(
        self,
        question: str,
        expected_answer: str,
        agent_answer: str,
        **kwargs,
    ) -> dict:
        return {
            "question": question,
            "expected_answer": expected_answer,
            "agent_answer": agent_answer,
        }

    def parse_response(self, response: str) -> JudgeResult:
        try:
            data = self._extract_json(response)
            has_contradiction = data.get("has_contradiction", False)
            severity = data.get("severity", "none")

            # Score: 1.0 if no contradiction, penalize based on severity
            severity_penalties = {
                "none": 0.0,
                "minor": 0.1,
                "major": 0.3,
                "critical": 0.5,
            }
            penalty = severity_penalties.get(severity, 0.2)
            score = 1.0 - penalty if has_contradiction else 1.0

            return JudgeResult(
                score=score,
                details={
                    "has_contradiction": has_contradiction,
                    "contradiction_type": data.get("contradiction_type"),
                    "severity": severity,
                    "contradictions_found": data.get("contradictions_found", []),
                    "reasoning": data.get("reasoning", ""),
                },
                raw_response=response,
            )
        except Exception as e:
            logger.warning(f"Failed to parse contradiction response: {e}")
            return JudgeResult(
                score=1.0,  # Assume no contradiction on parse failure
                details={"parse_error": str(e)},
                raw_response=response,
            )


class ProcessQualityJudge(BaseJudge):
    """Judge for evaluating the quality of the research process."""

    def get_prompt_name(self) -> str:
        return "process_quality_judge"

    def get_prompt_kwargs(
        self,
        question: str,
        tool_calls: list[dict],
        trajectory: str,
        agent_answer: str,
        **kwargs,
    ) -> dict:
        # Format tool calls for the prompt
        tool_calls_str = ""
        for i, tc in enumerate(tool_calls, 1):
            tool_calls_str += f"\n{i}. {tc.get('tool_name', 'unknown')}"
            if tc.get('arguments'):
                args_str = json.dumps(tc['arguments'], indent=2)
                tool_calls_str += f"\n   Arguments: {args_str}"
            if tc.get('result'):
                result_preview = str(tc['result'])[:500]
                tool_calls_str += f"\n   Result: {result_preview}..."

        return {
            "question": question,
            "tool_calls": tool_calls_str or "(No tool calls recorded)",
            "trajectory": trajectory,
            "agent_answer": agent_answer,
        }

    def parse_response(self, response: str) -> JudgeResult:
        try:
            data = self._extract_json(response)
            return JudgeResult(
                score=float(data.get("overall_score", 0.5)),
                details={
                    "dimension_scores": data.get("dimension_scores", {}),
                    "strengths": data.get("strengths", []),
                    "weaknesses": data.get("weaknesses", []),
                    "recommendations": data.get("recommendations", []),
                    "reasoning": data.get("reasoning", ""),
                },
                raw_response=response,
            )
        except Exception as e:
            logger.warning(f"Failed to parse process quality response: {e}")
            return JudgeResult(
                score=0.5,
                details={"parse_error": str(e)},
                raw_response=response,
            )


class EvaluationOrchestrator:
    """Orchestrates multiple judges to produce a final evaluation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        weights: Optional[dict[str, float]] = None,
    ):
        self.factual_judge = FactualAccuracyJudge(model=model)
        self.contradiction_judge = ContradictionJudge(model=model)
        self.process_judge = ProcessQualityJudge(model=model)

        # Default weights for aggregating scores
        self.weights = weights or {
            "factual_accuracy": 0.5,
            "contradiction": 0.2,
            "process_quality": 0.3,
        }

    def _build_factual_details(self, factual_result: JudgeResult) -> dict:
        """Build self-contained factual accuracy details for transparency."""
        rubric_scores = factual_result.details.get("rubric_scores", {})
        matched = sum(1 for v in rubric_scores.values() if v == 1)
        total = len(rubric_scores)

        return {
            "score": factual_result.score,
            "rubrics_matched": matched,
            "rubrics_total": total,
            "percentage": f"{(matched/total*100):.1f}%" if total > 0 else "0%",
            "rubric_scores": rubric_scores,
            "facts_found": factual_result.details.get("key_facts_found", []),
            "facts_missing": factual_result.details.get("key_facts_missing", []),
            "errors": factual_result.details.get("errors", []),
            "reasoning": factual_result.details.get("reasoning", "")
        }

    def _build_contradiction_details(self, contradiction_result: JudgeResult, expected_answer: str) -> dict:
        """Build self-contained contradiction details for transparency."""
        return {
            "has_contradiction": contradiction_result.details.get("has_contradiction", False),
            "severity": contradiction_result.details.get("severity", "none"),
            "score": contradiction_result.score,
            "expected_answer": expected_answer,
            "contradictions_found": contradiction_result.details.get("contradictions_found", []),
            "reasoning": contradiction_result.details.get("reasoning", "")
        }

    def _build_process_details(self, process_result: JudgeResult) -> dict:
        """Build self-contained process quality details for transparency."""
        return {
            "score": process_result.score,
            "dimension_scores": process_result.details.get("dimension_scores", {}),
            "strengths": process_result.details.get("strengths", []),
            "weaknesses": process_result.details.get("weaknesses", []),
            "recommendations": process_result.details.get("recommendations", []),
            "reasoning": process_result.details.get("reasoning", "")
        }

    async def evaluate(
        self,
        question: str,
        expert_answer: str,
        agent_answer: str,
        trajectory: str,
        tool_calls: list[dict],
        rubrics: list[dict] = None,
    ) -> dict[str, Any]:
        """Run all judges and aggregate results."""

        # Run judges concurrently
        import asyncio

        # Separate rubrics by operator type
        correctness_rubrics = []
        expected_answer_for_contradiction = expert_answer  # Default to expert_answer

        if rubrics:
            # Extract correctness rubrics (for FactualAccuracyJudge)
            correctness_rubrics = [
                r.get('criteria', '')
                for r in rubrics
                if r.get('operator') == 'correctness'
            ]

            # Extract contradiction rubric as expected answer (for ContradictionJudge)
            contradiction_rubrics = [
                r.get('criteria', '')
                for r in rubrics
                if r.get('operator') == 'contradiction'
            ]
            if contradiction_rubrics:
                # Use the contradiction rubric as the expected answer (semantic anchor)
                expected_answer_for_contradiction = contradiction_rubrics[0]

        factual_task = self.factual_judge.evaluate(
            question=question,
            expert_answer=expert_answer,
            agent_answer=agent_answer,
            correctness_rubrics=correctness_rubrics,
        )

        contradiction_task = self.contradiction_judge.evaluate(
            question=question,
            expected_answer=expected_answer_for_contradiction,
            agent_answer=agent_answer,
        )

        process_task = self.process_judge.evaluate(
            question=question,
            tool_calls=tool_calls,
            trajectory=trajectory,
            agent_answer=agent_answer,
        )

        factual_result, contradiction_result, process_result = await asyncio.gather(
            factual_task, contradiction_task, process_task
        )

        # Build self-contained details for transparency
        factual_details = self._build_factual_details(factual_result)
        contradiction_details = self._build_contradiction_details(contradiction_result, expected_answer_for_contradiction)
        process_details = self._build_process_details(process_result)

        # Calculate RL reward (with process quality)
        rl_reward = (
            self.weights["factual_accuracy"] * factual_result.score +
            self.weights["contradiction"] * contradiction_result.score +
            self.weights["process_quality"] * process_result.score
        )

        # Calculate BENCHMARK reward (paper-conform: only factual + contradiction)
        benchmark_reward = (
            0.7 * factual_result.score +
            0.3 * contradiction_result.score
        )

        # Pass criteria
        has_contradiction = contradiction_result.details.get("has_contradiction", False)
        benchmark_passed = (factual_result.score >= 0.8 and not has_contradiction)
        rl_passed = (rl_reward >= 0.8)

        # Sum up judge costs
        total_judge_cost = (
            factual_result.cost_usd +
            contradiction_result.cost_usd +
            process_result.cost_usd
        )

        # Emoji CLI output with pass/fail indicators
        benchmark_status = "✅ PASS" if benchmark_passed else "❌ FAIL"
        rl_status = "✅ PASS" if rl_passed else "❌ FAIL"

        logger.info(f"💰 Total judge cost: ${total_judge_cost:.6f} "
                   f"(factual: ${factual_result.cost_usd:.6f}, "
                   f"contradiction: ${contradiction_result.cost_usd:.6f}, "
                   f"process: ${process_result.cost_usd:.6f})")
        logger.info(f"📊 Benchmark: {benchmark_reward:.3f} {benchmark_status} "
                   f"(factual: {factual_result.score:.2f}, contradiction: {contradiction_result.score:.2f})")
        logger.info(f"🎯 RL Mode: {rl_reward:.3f} {rl_status} "
                   f"(factual: {factual_result.score:.2f}, contradiction: {contradiction_result.score:.2f}, "
                   f"process: {process_result.score:.2f})")

        return {
            # Legacy format (for backward compatibility) - uses RL reward
            "reward": rl_reward,
            "factual_accuracy": factual_result.score,
            "has_contradiction": has_contradiction,
            "contradiction_severity": contradiction_result.details.get("severity", "none"),
            "process_quality": process_result.score,
            "passed": rl_passed,

            # BENCHMARK MODE - Self-contained with full transparency
            "benchmark_mode": {
                "reward": benchmark_reward,
                "passed": benchmark_passed,
                "pass_threshold": 0.8,
                "pass_criteria": "factual_accuracy >= 0.8 AND has_contradiction == false",
                "formula": "0.7 * factual_accuracy + 0.3 * contradiction_score",
                "factual_accuracy": factual_details,
                "contradiction": contradiction_details,
            },

            # RL MODE - Self-contained with full transparency
            "rl_mode": {
                "reward": rl_reward,
                "passed": rl_passed,
                "pass_threshold": 0.8,
                "pass_criteria": "reward >= 0.8",
                "formula": "0.5 * factual + 0.2 * contradiction + 0.3 * process",
                "weights": self.weights,
                "factual_accuracy": factual_details,
                "contradiction": contradiction_details,
                "process_quality": process_details,
            },

            # Cost tracking
            "costs": {
                "total_usd": total_judge_cost,
                "judges_breakdown": {
                    "factual": factual_result.cost_usd,
                    "contradiction": contradiction_result.cost_usd,
                    "process": process_result.cost_usd,
                }
            },
        }


# Convenience function for simple evaluations
async def evaluate_answer(
    question: str,
    expert_answer: str,
    agent_answer: str,
    trajectory: str = "",
    tool_calls: list[dict] = None,
    rubrics: list[dict] = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Convenience function to run full evaluation."""
    orchestrator = EvaluationOrchestrator(model=model)
    return await orchestrator.evaluate(
        question=question,
        expert_answer=expert_answer,
        agent_answer=agent_answer,
        trajectory=trajectory,
        tool_calls=tool_calls or [],
        rubrics=rubrics,
    )
