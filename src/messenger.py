"""
Messenger for A2A agent communication.

Based on green-agent-template pattern.
Includes OpenTelemetry tracing for Phoenix observability.
"""
import json
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)

# Optional OpenTelemetry tracing with OpenInference semantic conventions
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    _tracer = trace.get_tracer("finance_evaluator.a2a")
    _HAS_OTEL = True

    # OpenInference semantic conventions for better Phoenix display
    # See: https://github.com/Arize-ai/openinference
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
except ImportError:
    _HAS_OTEL = False
    _tracer = None
    INPUT_VALUE = OUTPUT_VALUE = OPENINFERENCE_SPAN_KIND = None


DEFAULT_TIMEOUT = 300


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
):
    """Returns dict with context_id, response and status (if exists)"""
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs = {"response": "", "context_id": None, "artifacts": []}

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    outputs["artifacts"] = task.artifacts  # Store artifacts for structured data access
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs


class Messenger:
    """Helper class for A2A agent-to-agent communication."""

    def __init__(self, debug_mode: bool = False):
        self._context_ids = {}
        self.debug_mode = debug_mode
        self.current_trace = None  # Will be set per task

    def set_trace(self, trace):
        """Set the current task trace for A2A message logging."""
        self.current_trace = trace

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> tuple[str, list]:
        """
        Communicate with another agent.

        Args:
            message: The message to send
            url: The agent's URL endpoint
            new_conversation: If True, start fresh; if False, continue existing
            timeout: Timeout in seconds (default: 300)

        Returns:
            tuple: (response_text, artifacts) - text response and list of Artifact objects
        """
        # Log outbound A2A message if debugging enabled
        if self.debug_mode and self.current_trace:
            self.current_trace.add_a2a_message(
                direction="sent",
                message_type="agent_request",
                payload={
                    "url": url,
                    "message": message[:1000],  # Truncate for readability
                    "new_conversation": new_conversation,
                }
            )

        # Create OpenTelemetry span for A2A communication if available
        if _HAS_OTEL and _tracer:
            with _tracer.start_as_current_span(
                "A2A Agent Communication",
                kind=SpanKind.CLIENT,
                attributes={
                    OPENINFERENCE_SPAN_KIND: "CHAIN",  # Mark as chain for Phoenix
                    "a2a.target_url": url,
                    "a2a.new_conversation": new_conversation,
                    # Use OpenInference input.value for Phoenix display
                    INPUT_VALUE: message[:2000] if message else "",
                }
            ) as span:
                try:
                    outputs = await send_message(
                        message=message,
                        base_url=url,
                        context_id=None if new_conversation else self._context_ids.get(url, None),
                        timeout=timeout,
                    )
                    response = outputs.get("response", "")

                    # Use OpenInference output.value for Phoenix display
                    span.set_attribute(OUTPUT_VALUE, response[:2000] if response else "")
                    span.set_attribute("a2a.context_id", outputs.get("context_id", "") or "")
                    span.set_attribute("a2a.status", outputs.get("status", "completed"))

                    if outputs.get("status", "completed") != "completed":
                        span.set_status(Status(StatusCode.ERROR, f"Agent returned: {outputs.get('status')}"))
                        raise RuntimeError(f"{url} responded with: {outputs}")

                    span.set_status(Status(StatusCode.OK))
                    self._context_ids[url] = outputs.get("context_id", None)

                    # Log inbound A2A message if debugging enabled
                    if self.debug_mode and self.current_trace:
                        self.current_trace.add_a2a_message(
                            direction="received",
                            message_type="agent_response",
                            payload={
                                "url": url,
                                "response": response[:1000],  # Truncate
                                "context_id": outputs.get("context_id", ""),
                                "status": outputs.get("status", "completed"),
                            }
                        )

                    return response, outputs.get("artifacts", [])

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        else:
            # No tracing available
            outputs = await send_message(
                message=message,
                base_url=url,
                context_id=None if new_conversation else self._context_ids.get(url, None),
                timeout=timeout,
            )
            if outputs.get("status", "completed") != "completed":
                raise RuntimeError(f"{url} responded with: {outputs}")
            self._context_ids[url] = outputs.get("context_id", None)

            response = outputs["response"]

            # Log inbound A2A message if debugging enabled (non-tracing path)
            if self.debug_mode and self.current_trace:
                self.current_trace.add_a2a_message(
                    direction="received",
                    message_type="agent_response",
                    payload={
                        "url": url,
                        "response": response[:1000],  # Truncate
                        "context_id": outputs.get("context_id", ""),
                        "status": outputs.get("status", "completed"),
                    }
                )

            return response, outputs.get("artifacts", [])

    def reset(self):
        """Reset conversation contexts."""
        self._context_ids = {}
