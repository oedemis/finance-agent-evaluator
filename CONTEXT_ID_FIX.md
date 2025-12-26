# Context ID Mismatch Fix

## Problem

The green agent (evaluator) was registering the Gymnasium environment with `task.id` (e.g., "task_000"), but the purple agent (baseline) was calling MCP tools with the A2A conversation `context_id` (a UUID like "b68b0367-3e79-4373-8e35-304899084982").

This caused MCP tool calls to fail with:
```
RuntimeError: No environment registered for context: b68b0367-3e79-4373-8e35-304899084982
```

## Root Cause

1. Green agent registered environment BEFORE sending the first A2A message
2. At that time, the A2A `context_id` was not yet available
3. Green agent used `task.id` as a fallback
4. Purple agent received the A2A message with a proper `context_id` (set by A2A protocol)
5. Purple agent passed this `context_id` to MCP client
6. MCP server couldn't find the environment (registered under `task.id` instead)

## Solution

Changed the timing of environment registration:

**Before:**
- Create environment
- Register with `task.id`
- Send first A2A message
- Purple agent calls MCP tools (with A2A `context_id`) → **FAILS**

**After:**
- Create environment
- Send first A2A message
- Extract A2A `context_id` from messenger
- Register with A2A `context_id`
- Purple agent calls MCP tools (with same A2A `context_id`) → **SUCCESS**

## Changes Made

Modified `/Users/ooedemis/dev/benchmarks/finance-agent-evaluator/src/agent.py`:

1. Moved environment registration to AFTER first A2A message exchange
2. Extract `context_id` from `messenger._context_ids[agent_url]`
3. Register environment with the correct A2A `context_id`
4. Updated cleanup to only unregister if registration succeeded

## Testing

To verify the fix works:
1. Start MCP server: `python -m finance_agent_evaluator.src.mcp_server --port 9020`
2. Start purple agent: `python -m finance_agent_baseline.src.agent --port 9019`
3. Run evaluation: Should no longer see "No environment registered" errors

The logs should show:
```
INFO - Registered environment for task task_000 with A2A context_id: <uuid>
INFO - [<uuid>] MCP edgar_search → routing to environment
```

The context_id should match between registration and MCP tool calls.
