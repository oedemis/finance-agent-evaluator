FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy agentbeats library (must be provided in docker-compose context)
COPY agentbeats-src/ /agentbeats/

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

# Install dependencies (using modified path for Docker)
RUN uv pip install --system -e /agentbeats && \
    uv pip install --system gymnasium a2a-sdk pydantic aiohttp beautifulsoup4 python-dotenv uvicorn starlette openai pandas && \
    uv pip install --system -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9009/health || exit 1

# Expose A2A port
EXPOSE 9009

# Set entrypoint
ENTRYPOINT ["python", "src/evaluator.py"]

# Default arguments
CMD ["--host", "0.0.0.0", "--port", "9009"]
