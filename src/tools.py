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
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import aiohttp
from bs4 import BeautifulSoup
import litellm


# Tool costs (approximate)
TOOL_COSTS = {
    "edgar_search": 0.01,
    "google_web_search": 0.005,
    "parse_html_page": 0.001,
    "retrieve_information": 0.02,
    "submit_answer": 0.0,
}


def get_tool_definitions() -> list[dict]:
    """Get JSON schema definitions for all available tools."""
    return [
        {
            "name": "edgar_search",
            "description": "Search the EDGAR Database through the SEC API. Returns filing metadata (not full text).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword or phrase to search"
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of form types (e.g., ['10-K', '10-Q'])"
                    },
                    "ciks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of CIKs to filter by"
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
            "description": "Search the web for information using Google",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["search_query"]
            }
        },
        {
            "name": "parse_html_page",
            "description": "Parse an HTML page and save the content to data storage. Use 'key' to reference this content later with retrieve_information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the HTML page to parse"
                    },
                    "key": {
                        "type": "string",
                        "description": "The key to store the parsed content under"
                    }
                },
                "required": ["url", "key"]
            }
        },
        {
            "name": "retrieve_information",
            "description": "Retrieve and analyze information from stored documents using LLM. Your prompt MUST include {{key_name}} to reference stored documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt with {{key_name}} placeholders for stored documents"
                    },
                    "input_character_ranges": {
                        "type": "object",
                        "description": "Optional: character ranges for each key as {key: [start, end]}"
                    }
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "submit_answer",
            "description": "Submit your final answer to the question. This ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete answer to the research question"
                    }
                },
                "required": ["answer"]
            }
        }
    ]


class BaseTool(ABC):
    """Abstract base class for tools."""

    name: str
    cost: float = 0.0

    @abstractmethod
    async def execute(self, arguments: dict, **kwargs) -> tuple[str, float]:
        """Execute the tool. Returns (result_string, cost)."""
        pass


class EdgarSearchTool(BaseTool):
    """Search SEC EDGAR database."""

    name = "edgar_search"
    cost = TOOL_COSTS["edgar_search"]

    def __init__(self):
        self.api_key = os.getenv("SEC_EDGAR_API_KEY")
        self.api_url = "https://api.sec-api.io/full-text-search"

    async def execute(self, arguments: dict, **kwargs) -> tuple[str, float]:
        if not self.api_key:
            return json.dumps({"error": "SEC_EDGAR_API_KEY not set"}), 0.0

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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()

            top_n = int(arguments.get("top_n_results", 5))
            filings = result.get("filings", [])[:top_n]

            return json.dumps({"filings": filings, "total": len(filings)}), self.cost

        except Exception as e:
            return json.dumps({"error": str(e)}), self.cost


class GoogleSearchTool(BaseTool):
    """Search the web using Google via SerpAPI."""

    name = "google_web_search"
    cost = TOOL_COSTS["google_web_search"]

    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")

    async def execute(self, arguments: dict, **kwargs) -> tuple[str, float]:
        if not self.api_key:
            return json.dumps({"error": "SERPAPI_API_KEY not set"}), 0.0

        query = arguments.get("search_query", "")

        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": query,
            "num": 10,
        }

        try:
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

            return json.dumps({"results": simplified}), self.cost

        except Exception as e:
            return json.dumps({"error": str(e)}), self.cost


class ParseHtmlTool(BaseTool):
    """Parse HTML page and store content."""

    name = "parse_html_page"
    cost = TOOL_COSTS["parse_html_page"]

    def __init__(self):
        self.headers = {"User-Agent": "FinanceAgentBenchmark/2.0"}

    async def execute(
        self,
        arguments: dict,
        data_storage: Optional[dict] = None,
        **kwargs
    ) -> tuple[str, float]:
        url = arguments.get("url", "")
        key = arguments.get("key", "")

        if not url or not key:
            return json.dumps({"error": "url and key are required"}), 0.0

        if data_storage is None:
            data_storage = {}

        try:
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

            result = {
                "success": True,
                "key": key,
                "content_length": len(text),
                "stored_keys": list(data_storage.keys()),
            }

            return json.dumps(result), self.cost

        except Exception as e:
            return json.dumps({"error": str(e)}), self.cost


class RetrieveInformationTool(BaseTool):
    """Retrieve information from stored documents using LLM via LiteLLM."""

    name = "retrieve_information"
    cost = TOOL_COSTS["retrieve_information"]

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def execute(
        self,
        arguments: dict,
        data_storage: Optional[dict] = None,
        **kwargs
    ) -> tuple[str, float]:
        import re

        prompt = arguments.get("prompt", "")
        character_ranges = arguments.get("input_character_ranges", {}) or {}

        if data_storage is None:
            data_storage = {}

        # Validate prompt has placeholders
        if not re.search(r"{{[^{}]+}}", prompt):
            return json.dumps({
                "error": "Prompt must include {{key_name}} placeholders for stored documents"
            }), 0.0

        # Find all keys in the prompt
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        formatted_data = {}

        for key in keys:
            if key not in data_storage:
                return json.dumps({
                    "error": f"Key '{key}' not found in storage. Available: {list(data_storage.keys())}"
                }), 0.0

            content = data_storage[key]

            # Apply character range if specified
            if key in character_ranges:
                char_range = character_ranges[key]
                if len(char_range) == 2:
                    content = content[int(char_range[0]):int(char_range[1])]

            formatted_data[key] = content

        # Format prompt
        formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
        try:
            final_prompt = formatted_prompt.format(**formatted_data)
        except KeyError as e:
            return json.dumps({"error": f"Key error: {e}"}), 0.0

        # Call LLM via LiteLLM
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=1000,
            )
            answer = response.choices[0].message.content
            return json.dumps({"result": answer}), self.cost

        except Exception as e:
            return json.dumps({"error": str(e)}), self.cost


class ToolExecutor:
    """Executes tools by name."""

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
    ) -> tuple[str, float]:
        """Execute a tool synchronously."""
        import asyncio

        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}), 0.0

        tool = self._tools[tool_name]

        # Run async tool
        result, cost = asyncio.run(
            tool.execute(arguments, data_storage=data_storage)
        )

        return result, cost

    async def execute_async(
        self,
        tool_name: str,
        arguments: dict,
        data_storage: Optional[dict] = None,
    ) -> tuple[str, float]:
        """Execute a tool asynchronously."""
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}), 0.0

        tool = self._tools[tool_name]
        return await tool.execute(arguments, data_storage=data_storage)
