# Finance Agent Evaluator (Green Agent)

**Version:** 2.0.0
**Type:** Green Agent (Evaluator)
**Framework:** MCP + Gymnasium + A2A Protocol

## Overview

The Finance Agent Evaluator is a green agent that evaluates purple agents (financial research agents) on 537 real-world financial analysis tasks across 9 categories.

### Key Features

- **State-Tracked Research**: Gymnasium environment tracks research process :) trial
- **Multi-Dimensional Evaluation**: 6 metrics (factual accuracy, process quality, cost, time, etc.)
- **League of Judges**: 2 specialized LLM judges + rule-based metrics
- **Rich Reporting**: Class-balanced accuracy, dimensional scores, weakness profiling

## Architecture

The Finance Agent Evaluator runs **dual servers** for clean separation of concerns:

```
Green Agent (Dual Server Architecture):
├─ A2A Server (Port 9009)          # Task/conversation management
│  ├─ Orchestrator                 # Main coordinator
│  ├─ Gymnasium Environment        # Financial research simulation
│  ├─ Evaluation System            # Judges + Metrics
│  └─ Metrics Aggregator           # Results aggregation
│
└─ MCP Server (Port 9020)          # Tool exposure for purple agents
   ├─ edgar_search                 # SEC EDGAR database search
   ├─ google_web_search            # Web search via SerpAPI
   ├─ parse_html_page              # HTML content extraction
   ├─ retrieve_information         # LLM-based information retrieval
   └─ submit_answer                # Final answer submission
```

**Protocol Usage:**
- **A2A (Agent-to-Agent)**: Task assignment, progress tracking, results collection
- **MCP (Model Context Protocol)**: Tool discovery and execution

**Benefits:**
- Purple agents can use **any architecture** (no text format requirements)
- Purple agents discover tools via standard MCP protocol
- Clean separation: A2A for conversations, MCP for tools
- Supports 1000s of diverse purple agent implementations

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
# Run both A2A and MCP servers
uv run python src/server.py --host 127.0.0.1 --port 9009 --mcp-port 9020
```

**CLI Options:**
```
--host HOST           Host to bind (default: 127.0.0.1)
--port PORT           A2A server port (default: 9009)
--mcp-port PORT       MCP server port (default: 9020, 0 to disable)
--card-url URL        External URL for agent card
--data-path PATH      Path to dataset (default: data/public.csv)
--trace-dir DIR       Directory to save execution traces (default: traces)
--phoenix             Enable Phoenix observability
--phoenix-endpoint    Phoenix endpoint URL (default: http://localhost:6006)
--no-llm-judges       Disable LLM judges (use heuristic evaluation)
--judge-model MODEL   Model to use for LLM judges (default: gpt-4o-mini)
--debug-a2a           Log all A2A protocol messages to trace files
```

**Server Endpoints:**
- A2A Server: `http://localhost:9009/` - Task management
- MCP Server: `http://localhost:9020/mcp` - Tool access for purple agents
- Health Check: `http://localhost:9009/health`

### With Phoenix Observability

```bash
# Install Phoenix dependencies
pip install ".[phoenix]"

# Start Phoenix server (separate terminal)
phoenix serve

# Run with Phoenix tracing
uv run python src/server.py --phoenix --trace-dir ../FAB/traces
```

### Run with Docker

```bash
docker build -t finance-evaluator:v2.0 .
docker run -p 9009:9009 --env-file .env finance-evaluator:v2.0
```

### Run with FAB Scenario Runner in another repository

```bash
cd ../FAB
uv run fab-run scenario.toml
```

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

**LLM Judges** (`judges.py`):
- **FactualAccuracyJudge**: Rubric-based evaluation against expert answer
- **ContradictionJudge**: Detects factual contradictions between agent/expert
- **ProcessQualityJudge**: Evaluates research methodology and tool usage
- **EvaluationOrchestrator**: Runs all judges concurrently, aggregates results

**Prompts** (`prompts/`):
- `factual_accuracy_judge.txt`: Factual accuracy evaluation prompt
- `contradiction_judge.txt`: Contradiction detection prompt
- `process_quality_judge.txt`: Process quality evaluation prompt
- `task_prompt.txt`: Initial task prompt for purple agent

**Computed Metrics:**
- Cost Tracking: $ per task
- Time Tracking: Seconds per task
- Trajectory Stats: Tool usage, redundancy, coverage

### 4. MCP Tools (Exposed to Purple Agents)

The green agent exposes 5 tools via the MCP server that purple agents can discover and call:

1. **edgar_search** - Search SEC EDGAR database for company filings
   - Arguments: `query`, `form_types` (optional), `ciks` (optional), `start_date` (optional), `end_date` (optional), `page`, `top_n_results`
   - Returns: Search results with filing URLs

2. **google_web_search** - Search the web using Google (SerpAPI)
   - Arguments: `search_query`
   - Returns: Web search results with URLs

3. **parse_html_page** - Parse HTML content from a URL and store for later retrieval
   - Arguments: `url`, `key` (storage key for later reference)
   - Returns: Success confirmation

4. **retrieve_information** - Extract information from stored documents using LLM
   - Arguments: `prompt` (use `{{key}}` placeholders to reference stored content)
   - Returns: Extracted information

5. **submit_answer** - Submit final answer (ends the task)
   - Arguments: `answer` (comprehensive answer text), `sources` (list of `{"url": "...", "name": "..."}`)
   - Returns: Submission confirmation

**State Management:**
- Each purple agent conversation gets isolated storage via `context_id`
- `parse_html_page` stores content, `retrieve_information` retrieves it
- State persists across tool calls within a single task

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
│   ├── server.py             # Main entry point (A2A + MCP servers)
│   ├── mcp_server.py         # MCP server (tool exposure via FastMCP)
│   ├── executor.py           # AgentExecutor (manages agent instances)
│   ├── agent.py              # FinanceEvaluatorAgent (evaluation logic)
│   ├── messenger.py          # A2A client for purple agent communication
│   ├── environment.py        # Gymnasium environment
│   ├── dataset.py            # Dataset loading utilities
│   ├── tools.py              # Tool implementations (EDGAR, Google, etc.)
│   ├── judges.py             # LLM-based evaluation judges
│   ├── tracer.py             # Execution trace logging
│   ├── observability.py      # Phoenix integration
│   └── prompts/              # Prompt templates (separated from code)
│       ├── __init__.py       # load_prompt(), format_prompt()
│       ├── factual_accuracy_judge.txt
│       ├── contradiction_judge.txt
│       ├── process_quality_judge.txt
│       └── task_prompt.txt
├── data/
│   └── public.csv            # Public dataset (50 tasks)
├── tests/
│   └── ...
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


## References

- [AgentBeats Documentation](https://docs.agentbeats.dev/)
- [Finance Agent Paper](https://arxiv.org/abs/2508.00828)

## Contact

For questions or issues, please open an issue in the repository.
