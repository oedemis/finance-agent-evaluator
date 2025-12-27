FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/ ./data/

# Install dependencies
RUN uv pip install --system -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9009/health || exit 1

# Expose A2A port
EXPOSE 9009

# Set entrypoint - using new server.py
ENTRYPOINT ["python", "src/server.py"]

# Default arguments
CMD ["--host", "0.0.0.0", "--port", "9009"]
