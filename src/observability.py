"""
Phoenix Observability Integration for Finance Agent Benchmark.

Provides LLM tracing via Arize Phoenix running in Docker.
Uses OpenInference instrumentation for comprehensive tracing including tool calls.

IMPORTANT: Phoenix requires a running Docker container:
  docker run -d -p 6006:6006 arizephoenix/phoenix

Without Docker Phoenix running, the --phoenix flag will be ignored
and the benchmark will continue with JSON traces only.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("finance_evaluator.observability")

# Track if LiteLLM has been instrumented
_litellm_instrumented = False


def _check_phoenix_otlp_ready(endpoint: str, timeout: float = 2.0) -> bool:
    """
    Check if Phoenix OTLP endpoint is ready to receive traces.

    Phoenix OTLP HTTP endpoint is at {base}/v1/traces.
    """
    import httpx

    # Ensure endpoint has /v1/traces path
    if not endpoint.endswith("/v1/traces"):
        otlp_url = endpoint.rstrip('/') + "/v1/traces"
    else:
        otlp_url = endpoint

    try:
        # Send minimal OTLP request to check if endpoint accepts traces
        response = httpx.post(
            otlp_url,
            content=b"",
            headers={"Content-Type": "application/x-protobuf"},
            timeout=timeout,
        )

        # Phoenix OTLP returns:
        # - 200 on success
        # - 400 on invalid protobuf (endpoint exists and works)
        # - 415 if content type issue (endpoint exists)
        # - 405 means endpoint doesn't accept POST (not Phoenix OTLP)
        if response.status_code in (200, 400, 415):
            logger.debug(f"Phoenix OTLP ready at {otlp_url} (status: {response.status_code})")
            return True
        else:
            logger.debug(f"Phoenix OTLP check failed: status {response.status_code}")
            return False

    except httpx.ConnectError:
        logger.debug(f"Cannot connect to {otlp_url} - Phoenix not running")
        return False
    except httpx.TimeoutException:
        logger.debug(f"Timeout connecting to {otlp_url}")
        return False
    except Exception as e:
        logger.debug(f"Phoenix check error: {e}")
        return False


class PhoenixObservability:
    """
    Manages Phoenix observability integration.

    Only works with Phoenix running in Docker:
        docker run -d -p 6006:6006 arizephoenix/phoenix
    """

    def __init__(
        self,
        project_name: str = "finance-agent-benchmark",
        endpoint: Optional[str] = None,
    ):
        self.project_name = project_name
        # Base endpoint (without /v1/traces)
        base_endpoint = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        # Store both base and full OTLP endpoint
        self.base_endpoint = base_endpoint.rstrip('/').replace('/v1/traces', '')
        self.otlp_endpoint = self.base_endpoint + "/v1/traces"
        self._initialized = False
        self._tracer_provider = None

    def initialize(self) -> bool:
        """
        Initialize Phoenix tracing if Docker Phoenix is running.

        Returns True if Phoenix is available and tracing is enabled.
        Returns False (gracefully) if Phoenix is not available.
        """
        if self._initialized:
            return True

        logger.info(f"Checking for Phoenix at {self.base_endpoint}...")

        # Check OTLP endpoint with full path
        if not _check_phoenix_otlp_ready(self.otlp_endpoint):
            logger.warning("Phoenix not running or OTLP endpoint not ready")
            logger.info("")
            logger.info("To enable Phoenix tracing:")
            logger.info("  1. Start Phoenix: docker run -d -p 6006:6006 arizephoenix/phoenix")
            logger.info("  2. Run with --phoenix flag")
            logger.info("")
            logger.info("Continuing without Phoenix (JSON traces still saved)")
            return False

        try:
            # Only import after confirming Phoenix is available
            from phoenix.otel import register

            # IMPORTANT: Use full endpoint path with /v1/traces
            # See: https://github.com/Arize-ai/phoenix/issues/4529
            self._tracer_provider = register(
                project_name=self.project_name,
                endpoint=self.otlp_endpoint,  # Full path: http://localhost:6006/v1/traces
                auto_instrument=True,
            )

            # Also instrument LiteLLM with OpenInference for comprehensive tracing
            global _litellm_instrumented
            if not _litellm_instrumented:
                try:
                    from openinference.instrumentation.litellm import LiteLLMInstrumentor
                    LiteLLMInstrumentor().instrument(tracer_provider=self._tracer_provider)
                    _litellm_instrumented = True
                    logger.info("LiteLLM OpenInference instrumentation enabled")
                except ImportError:
                    logger.debug("OpenInference LiteLLM instrumentation not available")
                    # Fall back to callback approach
                    try:
                        import litellm
                        if "arize_phoenix" not in (litellm.callbacks or []):
                            litellm.callbacks = litellm.callbacks or []
                            litellm.callbacks.append("arize_phoenix")
                            logger.info("LiteLLM arize_phoenix callback enabled")
                    except Exception as cb_err:
                        logger.debug(f"Could not set LiteLLM callback: {cb_err}")

            self._initialized = True
            logger.info(f"Phoenix tracing enabled!")
            logger.info(f"View traces at: {self.base_endpoint}")
            return True

        except ImportError as e:
            logger.warning(f"Phoenix package not installed: {e}")
            logger.info("Install: cd ../finance-agent-evaluator && uv sync --extra phoenix")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize Phoenix: {e}")
            return False

    def shutdown(self):
        """Shutdown tracing and flush remaining spans."""
        if not self._initialized:
            return
        try:
            if self._tracer_provider:
                if hasattr(self._tracer_provider, 'force_flush'):
                    self._tracer_provider.force_flush()
                if hasattr(self._tracer_provider, 'shutdown'):
                    self._tracer_provider.shutdown()
            self._initialized = False
            logger.debug("Phoenix tracing shutdown")
        except Exception:
            pass


def setup_observability(
    project_name: str = "finance-agent-benchmark",
    endpoint: Optional[str] = None,
) -> Optional[PhoenixObservability]:
    """
    Set up Phoenix observability if Docker Phoenix is running.

    Usage:
        1. Start Phoenix: docker run -d -p 6006:6006 arizephoenix/phoenix
        2. Run: uv run fab-run --phoenix
        3. View: http://localhost:6006

    If Phoenix is not running, returns None and benchmark continues
    with JSON traces only.
    """
    obs = PhoenixObservability(project_name=project_name, endpoint=endpoint)
    if obs.initialize():
        return obs
    return None
