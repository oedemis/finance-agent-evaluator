"""TDD: MCP tool descriptions must be clear without workflow hints."""
import pytest

# Test data: tool descriptions we want to validate
# We'll read actual MCP tool data, but for now let's define expectations

def test_task_prompt_is_minimal():
    """task_prompt should ONLY contain the question - no workflow hints."""
    from src.prompts import load_prompt
    
    prompt = load_prompt("task_prompt")
    
    # Should only have {question} placeholder
    assert "{question}" in prompt
    # Should NOT have workflow instructions
    forbidden_words = ["first", "before", "after", "always", "must", "should", "requirement"]
    for word in forbidden_words:
        assert word.lower() not in prompt.lower(), f"task_prompt should not contain '{word}'"
    
    # Should be very short
    assert len(prompt) < 100, "task_prompt should be minimal"

pytest.main([__file__, "-v"])
