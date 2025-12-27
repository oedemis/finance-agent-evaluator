"""
Tool implementations for Finance Agent Benchmark.

Tools available to purple agents:
1. edgar_search - Search SEC EDGAR database
2. google_web_search - Search the web via Google
3. parse_html_page - Parse and store HTML content
4. retrieve_information - Extract info from stored documents using LLM
5. submit_answer - Submit final answer (terminates episode)

Uses LiteLLM for LLM calls to support multiple providers.
"""
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Optional

import aiohttp
import backoff
from bs4 import BeautifulSoup
import litellm
from litellm import completion_cost

logger = logging.getLogger("finance_evaluator.tools")

# Tool costs (approximate)
TOOL_COSTS = {
    "edgar_search": 0.01,
    "google_web_search": 0.005,
    "parse_html_page": 0.001,
    "retrieve_information": 0.02,
    "submit_answer": 0.0,
}


def is_429(exception):
    """Check if exception is a 429 rate limit error."""
    is429 = (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
        or "429" in str(exception)
    )
    if is429:
        logger.error(f"429 error: {exception}")
    return is429


def retry_on_429(func):
    """Retry decorator for 429 rate limit errors."""
    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        max_tries=8,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_429(e),
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


def get_tool_definitions() -> list[dict]:
    """Get JSON schema definitions for all available tools."""
    return [
        {
            "name": "edgar_search",
            "description": "Search SEC EDGAR database for official regulatory filings. CRITICAL for: mergers/acquisitions (8-K), quarterly/annual financials (10-Q/10-K), executive changes, material events. Returns filing metadata with links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords related to topics, events, or financial terms (e.g., 'merger', 'acquisition', 'revenue', 'restructuring', 'dividend')"
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filing types: 10-K (annual), 10-Q (quarterly), 8-K (material events), DEF 14A (proxy), S-1 (IPO)"
                    },
                    "ciks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Company CIK numbers - 10-digit identifier padded with zeros (e.g., ['0001318605'] for Tesla, ['0000789019'] for Microsoft). Search company name on SEC.gov to find CIK."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in yyyy-mm-dd format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in yyyy-mm-dd format"
                    },
                    "page": {
                        "type": "string",
                        "description": "Page number for pagination"
                    },
                    "top_n_results": {
                        "type": "integer",
                        "description": "Number of results to return"
                    }
                },
                "required": ["query", "form_types", "ciks", "start_date", "end_date", "page", "top_n_results"]
            }
        },
        {
            "name": "google_web_search",
            "description": "Search the web for news, analysis, and company announcements. Use for recent events, market analysis, and general research. TIP: Include specific company names, dates, and key terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Search query with company name, topic, and date range"
                    }
                },
                "required": ["search_query"]
            }
        },
        {
            "name": "parse_html_page",
            "description": "Parse a webpage and store its content for later analysis. IMPORTANT: Parse MULTIPLE sources (3-4) to cross-reference information. Use meaningful keys like 'sec_filing_1', 'news_article_1'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the page to parse"
                    },
                    "key": {
                        "type": "string",
                        "description": "Unique key to store content (e.g., 'source_1', 'sec_8k_filing')"
                    }
                },
                "required": ["url", "key"]
            }
        },
        {
            "name": "retrieve_information",
            "description": "Analyze and synthesize information from stored documents using LLM. MUST use {{key_name}} placeholders. TIP: Combine multiple sources in one call: 'Summarize {{source_1}} {{source_2}} {{source_3}}'",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Analysis prompt with {{key_name}} placeholders for each stored document"
                    },
                    "input_character_ranges": {
                        "type": "object",
                        "description": "Optional: character ranges for each key as {key: [start, end]}"
                    }
                },
                "required": ["prompt"]
            }
        }
    ]


class BaseTool(ABC):
    """Abstract base class for tools (matches finance-agent architecture)."""

    name: str
    cost: float = 0.0

    @abstractmethod
    async def call_tool(self, arguments: dict, **kwargs) -> Any:
        """Execute the tool and return result data."""
        pass

    async def __call__(self, arguments: dict = None, **kwargs) -> dict:
        """
        Execute tool with error handling.
        Returns {"success": True/False, "result": ...} like finance-agent.
        """
        logger.info(f"[TOOL: {self.name.upper()}] Calling with arguments: {arguments}")

        try:
            result = await self.call_tool(arguments, **kwargs)
            logger.info(f"[TOOL: {self.name.upper()}] Returned: {str(result)[:200]}...")

            # Special handling for retrieve_information (returns usage metadata)
            # Note: usage is logged server-side, no need to return to client
            if self.name == "retrieve_information" and isinstance(result, dict):
                return {
                    "success": True,
                    "result": result.get("retrieval", result),
                }
            else:
                return {"success": True, "result": json.dumps(result) if not isinstance(result, str) else result}

        except Exception as e:
            is_verbose = os.environ.get("FINANCE_AGENT_VERBOSE", "0") == "1"
            error_msg = str(e)
            if is_verbose:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
                logger.warning(f"[TOOL: {self.name.upper()}] Error: {e}\nTraceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[TOOL: {self.name.upper()}] Error: {e}")

            return {"success": False, "result": error_msg}


class EdgarSearchTool(BaseTool):
    """Search SEC EDGAR database."""

    name = "edgar_search"
    cost = TOOL_COSTS["edgar_search"]

    def __init__(self):
        self.api_key = os.getenv("SEC_EDGAR_API_KEY")
        self.api_url = "https://api.sec-api.io/full-text-search"

    @retry_on_429
    async def call_tool(self, arguments: dict, **kwargs) -> list[dict]:
        if not self.api_key:
            raise ValueError("SEC_EDGAR_API_KEY not set")

        payload = {
            "query": arguments.get("query", ""),
            "formTypes": arguments.get("form_types", []),
            "ciks": arguments.get("ciks", []),
            "startDate": arguments.get("start_date", ""),
            "endDate": arguments.get("end_date", ""),
            "page": arguments.get("page", "1"),
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()

        top_n = int(arguments.get("top_n_results", 5))
        filings = result.get("filings", [])[:top_n]

        return filings


class GoogleSearchTool(BaseTool):
    """Search the web using Google via SerpAPI."""

    name = "google_web_search"
    cost = TOOL_COSTS["google_web_search"]

    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")

    @retry_on_429
    async def call_tool(self, arguments: dict, **kwargs) -> list[dict]:
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not set")

        query = arguments.get("search_query", "")

        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "num": 10,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search.json", params=params) as response:
                response.raise_for_status()
                result = await response.json()

        organic_results = result.get("organic_results", [])
        simplified = [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in organic_results[:10]
        ]

        return simplified


class ParseHtmlTool(BaseTool):
    """Parse HTML page and store content."""

    name = "parse_html_page"
    cost = TOOL_COSTS["parse_html_page"]

    def __init__(self):
        self.headers = {"User-Agent": "FinanceAgentBenchmark/2.0"}

    @retry_on_429
    async def call_tool(
        self,
        arguments: dict,
        data_storage: Optional[dict] = None,
        **kwargs
    ) -> str:
        url = arguments.get("url", "")
        key = arguments.get("key", "")

        if not url or not key:
            raise ValueError("url and key are required")

        if data_storage is None:
            data_storage = {}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, timeout=60) as response:
                response.raise_for_status()
                html_content = await response.text()

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.extract()

        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        # Store in data_storage
        data_storage[key] = text

        # Return success message like finance-agent
        tool_result = ""
        if key in list(data_storage.keys())[:-1]:  # Was already there (excluding just-added)
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"

        tool_result += f"SUCCESS: The result has been saved to the data storage under the key: {key}.\n"
        keys_list = "\n".join(data_storage.keys())
        tool_result += f"The data_storage currently contains the following keys:\n{keys_list}\n"

        return tool_result


class RetrieveInformationTool(BaseTool):
    """Retrieve information from stored documents using LLM via LiteLLM."""

    name = "retrieve_information"
    cost = TOOL_COSTS["retrieve_information"]

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model

    async def call_tool(
        self,
        arguments: dict,
        data_storage: Optional[dict] = None,
        **kwargs
    ) -> dict:
        import re

        prompt = arguments.get("prompt", "")
        character_ranges = arguments.get("input_character_ranges", {}) or {}

        if data_storage is None:
            data_storage = {}

        # Validate prompt has placeholders
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. "
                "Please try again with the correct format."
            )

        # Find all keys in the prompt
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}

        for key in keys:
            if key not in data_storage:
                available_keys = ', '.join(data_storage.keys())
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. "
                    f"Available keys are: {available_keys}"
                )

            content = data_storage[key]

            # Apply character range if specified
            if key in character_ranges:
                char_range = character_ranges[key]
                if len(char_range) == 0:
                    formatted_data[key] = content
                elif len(char_range) != 2:
                    raise ValueError(
                        f"ERROR: The character range for key '{key}' must be a list with two elements "
                        "or an empty list. Please try again with the correct format."
                    )
                else:
                    start_idx = int(char_range[0])
                    end_idx = int(char_range[1])
                    formatted_data[key] = content[start_idx:end_idx]
            else:
                # Use the full document if no range is specified
                formatted_data[key] = content

        # Format prompt
        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
        final_prompt = formatted_prompt.format(**formatted_data)

        # Call LLM via LiteLLM with concise response requirement
        # Keep responses small to avoid FastMCP Client StreamableHTTP bug with large payloads
        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": "Provide concise answers in 2-3 bullet points. Maximum 100 words. Be direct and factual."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=200,  # Reduced to keep responses consistently under 1000 bytes for FastMCP
        )

        answer = response.choices[0].message.content

        # Calculate ACTUAL cost from LiteLLM
        actual_cost = 0.0
        try:
            actual_cost = completion_cost(completion_response=response)
            logger.info(f"retrieve_information actual cost: ${actual_cost:.6f} (model: {self.model})")
        except Exception as e:
            logger.warning(f"Failed to calculate cost: {e}, using default")
            actual_cost = TOOL_COSTS["retrieve_information"]  # Fallback to fixed cost

        # Return with usage metadata like finance-agent
        usage_data = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "cost_usd": actual_cost,  # Add actual cost to usage data
        }

        return {
            "retrieval": answer,
            "usage": usage_data,
        }


class ToolExecutor:
    """Executes tools by name (matches finance-agent architecture)."""

    def __init__(self, retrieval_model: str = "gpt-4o-mini"):
        self._tools: dict[str, BaseTool] = {
            "edgar_search": EdgarSearchTool(),
            "google_web_search": GoogleSearchTool(),
            "parse_html_page": ParseHtmlTool(),
            "retrieve_information": RetrieveInformationTool(model=retrieval_model),
        }

    def execute(
        self,
        tool_name: str,
        arguments: dict,
        data_storage: Optional[dict] = None,
    ) -> tuple[dict, float]:
        """
        Execute a tool synchronously.
        Returns ({"success": bool, "result": ...}, cost).
        """
        import asyncio

        if tool_name not in self._tools:
            return {"success": False, "result": f"Unknown tool: {tool_name}"}, 0.0

        tool = self._tools[tool_name]

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - can't use asyncio.run()
            # Create a new thread to run the coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    tool(arguments, data_storage=data_storage)
                )
                result_dict = future.result()
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            result_dict = asyncio.run(
                tool(arguments, data_storage=data_storage)
            )

        return result_dict, tool.cost

    async def execute_async(
        self,
        tool_name: str,
        arguments: dict,
        data_storage: Optional[dict] = None,
    ) -> tuple[dict, float]:
        """
        Execute a tool asynchronously.
        Returns ({"success": bool, "result": ...}, cost).
        """
        if tool_name not in self._tools:
            return {"success": False, "result": f"Unknown tool: {tool_name}"}, 0.0

        tool = self._tools[tool_name]
        result_dict = await tool(arguments, data_storage=data_storage)
        return result_dict, tool.cost
