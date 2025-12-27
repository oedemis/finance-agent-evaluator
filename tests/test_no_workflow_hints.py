"""Test that MCP tools don't contain workflow hints (anti-reward-hacking)."""
import re

def test_parse_html_no_workflow_hints():
    """parse_html_page should not tell agents when to use it."""
    with open("src/mcp_server.py") as f:
        content = f.read()
    
    # Find parse_html_page docstring
    match = re.search(r'@mcp\.tool\(\)\nasync def parse_html_page.*?"""(.+?)"""', content, re.DOTALL)
    assert match, "Could not find parse_html_page docstring"
    
    docstring = match.group(1)
    
    # Should NOT mention other tools or workflow
    assert "retrieve_information" not in docstring, "Should not reference other tools"
    assert "later" not in docstring.lower(), "Should not imply workflow order"
    assert "after" not in docstring.lower(), "Should not imply workflow order"
    assert "before" not in docstring.lower(), "Should not imply workflow order"

def test_retrieve_information_no_workflow_hints():
    """retrieve_information should explain {{key}} without prescribing workflow."""
    with open("src/mcp_server.py") as f:
        content = f.read()
    
    match = re.search(r'@mcp\.tool\(\)\nasync def retrieve_information.*?"""(.+?)"""', content, re.DOTALL)
    assert match, "Could not find retrieve_information docstring"
    
    docstring = match.group(1)
    
    # Should have example but not workflow
    assert "{{" in docstring and "}}" in docstring, "Should show placeholder format"
    assert "Example:" in docstring or "example:" in docstring.lower(), "Should have example"
    # Should NOT say "from parse_html_page" - that's workflow
    assert "from parse_html_page" not in docstring, "Should not prescribe where keys come from"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
