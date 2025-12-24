"""
Dataset loader for Finance Agent Benchmark.

Loads tasks from public.csv and provides filtering by category.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RubricItem:
    """A single rubric item for evaluation."""
    operator: str  # 'correctness' or 'contradiction'
    criteria: str


@dataclass
class Task:
    """A single financial research task."""
    id: str
    question: str
    expert_answer: str
    category: str
    expert_time_mins: float
    rubrics: list[RubricItem]

    @property
    def difficulty(self) -> str:
        """Derive difficulty from category."""
        easy_categories = ["Quantitative Retrieval", "Qualitative Retrieval", "Numerical Reasoning"]
        medium_categories = ["Complex Retrieval", "Adjustments", "Beat or Miss"]
        hard_categories = ["Trends", "Financial Modeling", "Market Analysis"]

        if self.category in easy_categories:
            return "Easy"
        elif self.category in medium_categories:
            return "Medium"
        else:
            return "Hard"

    @property
    def correctness_rubrics(self) -> list[RubricItem]:
        """Get only correctness rubrics."""
        return [r for r in self.rubrics if r.operator == "correctness"]

    @property
    def contradiction_rubric(self) -> Optional[RubricItem]:
        """Get the contradiction rubric if exists."""
        for r in self.rubrics:
            if r.operator == "contradiction":
                return r
        return None


class DatasetLoader:
    """Loads and manages the Finance Agent Benchmark dataset."""

    CATEGORIES = [
        "Quantitative Retrieval",
        "Qualitative Retrieval",
        "Numerical Reasoning",
        "Complex Retrieval",
        "Adjustments",
        "Beat or Miss",
        "Trends",
        "Financial Modeling",
        "Market Analysis",
    ]

    def __init__(self, data_path: str = "data/public.csv"):
        self.data_path = Path(data_path)
        self._tasks: list[Task] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load tasks from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Parse rubrics from JSON string
                rubrics_str = row.get("Rubric", "[]")
                try:
                    # Handle single quotes in JSON
                    rubrics_str = rubrics_str.replace("'", '"')
                    rubrics_data = json.loads(rubrics_str)
                except json.JSONDecodeError:
                    rubrics_data = []

                rubrics = [
                    RubricItem(
                        operator=r.get("operator", ""),
                        criteria=r.get("criteria", "")
                    )
                    for r in rubrics_data
                ]

                # Parse expert time
                try:
                    expert_time = float(row.get("Expert time (mins)", 0))
                except (ValueError, TypeError):
                    expert_time = 0.0

                task = Task(
                    id=f"task_{idx:03d}",
                    question=row.get("Question", ""),
                    expert_answer=row.get("Answer", ""),
                    category=row.get("Question Type", "Unknown"),
                    expert_time_mins=expert_time,
                    rubrics=rubrics,
                )
                self._tasks.append(task)

    def get_tasks(
        self,
        categories: Optional[list[str]] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Task]:
        """
        Get tasks with optional filtering.

        Args:
            categories: List of categories to include. Use ["all"] for all categories.
            difficulty: Filter by difficulty ("Easy", "Medium", "Hard")
            limit: Maximum number of tasks to return

        Returns:
            List of matching tasks
        """
        tasks = self._tasks

        # Filter by categories
        if categories and "all" not in categories:
            tasks = [t for t in tasks if t.category in categories]

        # Filter by difficulty
        if difficulty:
            tasks = [t for t in tasks if t.difficulty == difficulty]

        # Apply limit
        if limit:
            tasks = tasks[:limit]

        return tasks

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def get_category_counts(self) -> dict[str, int]:
        """Get count of tasks per category."""
        counts = {}
        for task in self._tasks:
            counts[task.category] = counts.get(task.category, 0) + 1
        return counts

    @property
    def total_tasks(self) -> int:
        """Total number of tasks in dataset."""
        return len(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)
