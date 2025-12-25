"""
Prompt templates for Finance Agent Benchmark.

All prompts are stored as separate files for better maintainability.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from file.

    Args:
        name: Name of the prompt file (without .txt extension)

    Returns:
        The prompt template as a string
    """
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def format_prompt(name: str, **kwargs) -> str:
    """Load and format a prompt template.

    Args:
        name: Name of the prompt file (without .txt extension)
        **kwargs: Variables to substitute in the template

    Returns:
        The formatted prompt string
    """
    template = load_prompt(name)
    return template.format(**kwargs)
