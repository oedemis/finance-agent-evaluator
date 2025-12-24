# Finance Agent Evaluator (Green Agent)

**Version:** 2.0.0
**Type:** Green Agent (Evaluator)
**Framework:** Gymnasium + A2A Protocol

## Overview

The Finance Agent Evaluator is a green agent that evaluates purple agents (financial research agents) on 537 real-world financial analysis tasks across 9 categories.

### Key Features

- **State-Tracked Research**: Gymnasium environment tracks research process
- **Multi-Dimensional Evaluation**: 6 metrics (factual accuracy, process quality, cost, time, etc.)
- **League of Judges**: 2 specialized LLM judges + rule-based metrics
- **Rich Reporting**: Class-balanced accuracy, dimensional scores, weakness profiling

## Architecture

```
Green Agent Components:
├─ Orchestrator          # Main coordinator
├─ Gymnasium Environment # Financial research simulation
├─ Evaluation System     # Judges + Metrics
├─ Tools                 # EDGAR, Google Search, HTML Parser
└─ Metrics Aggregator    # Results aggregation
```

## Installation

### Prerequisites

- Python 3.10+
- uv (recommended) or pip

### Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .

# For development
uv sync --extra dev
```

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_key        # For LLM judges
SERPAPI_API_KEY=your_serpapi_key      # For Google Search
SEC_EDGAR_API_KEY=your_sec_key        # For EDGAR search
```

## Usage

### Run Locally

```bash
uv run python src/evaluator.py --host 0.0.0.0 --port 9009
```

### Run with Docker

```bash
docker build -t finance-evaluator:v2.0 .
docker run -p 9009:9009 --env-file .env finance-evaluator:v2.0
```

### Test with AgentBeats

See `../FAB/scenario.toml` for configuration.

## Components

### 1. Orchestrator
- Loads tasks from dataset
- Creates Gymnasium environment per task
- Communicates with purple agent via A2A
- Invokes evaluation system
- Reports results

### 2. Gymnasium Environment
- **State**: Research progress, resource usage
- **Actions**: Tool calls (EDGAR, Google, Parse, Retrieve)
- **Observations**: Tool results, error messages
- **Rewards**: Multi-dimensional (factual + process)

### 3. Evaluation System

**Judges (LLM-based):**
- Factual Accuracy Judge: Rubric-based evaluation
- Contradiction Judge: Detects factual contradictions

**Metrics (Computed):**
- Process Quality: Trajectory analysis (rule-based)
- Cost Tracking: $ per task
- Time Tracking: Seconds per task

### 4. Tools
- `search_edgar`: SEC EDGAR database search
- `search_google`: Web search via SerpAPI
- `parse_html`: HTML content extraction
- `retrieve_information`: LLM-based information retrieval

## Dataset

**Location:** `data/public.csv`
**Size:** 50 public tasks (537 total, rest private)
**Categories:** 9 (Quantitative Retrieval, Financial Modeling, etc.)

**Format:**
```csv
Question,Answer,Question Type,Expert time (mins),Rubric
```

## Evaluation Metrics

### Per-Task Metrics
- Reward (0-1): Aggregated score
- Factual Accuracy (0-1): Rubric pass rate
- Process Quality (0-1): Trajectory analysis
- Cost ($): Total cost
- Time (s): Total duration

### Aggregate Metrics
- Naive Accuracy: Overall pass rate
- Class-Balanced Accuracy: Average across categories
- Dimensional Scores: Per-dimension averages
- Weakness Profile: Identified weak areas

## Development

### Project Structure

```
finance-agent-evaluator/
├── src/
│   ├── evaluator.py          # Main green agent
│   ├── environment.py        # Gymnasium environment
│   ├── judges.py             # LLM judges
│   ├── metrics.py            # Metrics computation
│   ├── orchestrator.py       # Orchestration logic
│   ├── models.py             # Data models
│   └── tools/
│       ├── edgar.py          # EDGAR search
│       ├── google.py         # Google search
│       ├── html_parser.py    # HTML parsing
│       └── retriever.py      # Information retrieval
├── data/
│   └── public.csv            # Public dataset
├── tests/
│   ├── test_environment.py
│   ├── test_judges.py
│   └── test_tools.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format
uv run black src/

# Lint
uv run ruff check src/
```

## API Reference

### A2A Endpoint

**POST** `/agent`

Request body (EvalRequest):
```json
{
  "participants": {
    "agent": "http://purple-agent-url:port"
  },
  "config": {
    "num_tasks": 50,
    "categories": ["all"],
    "max_steps": 50,
    "timeout": 600
  }
}
```

Response: Streamed A2A messages with task updates and final artifact.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run tests and linting
5. Submit pull request

## License

TBD

## References

- [Design Document](../FAB/docs/design.md)
- [AgentBeats Documentation](https://docs.agentbeats.dev/)
- [Finance Agent Paper](https://arxiv.org/abs/2508.00828)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Contact

For questions or issues, please open an issue in the repository.
