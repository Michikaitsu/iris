"""
Tests for NSFW Filter Service
"""
import pytest
from src.api.services.nsfw_filter import check_nsfw_prompt, normalize_prompt


class TestNSFWFilter:
    """Test cases for NSFW content filtering"""
    
    def test_safe_prompt_passes(self):
        """Safe prompts should not be blocked"""
        safe_prompts = [
            "anime girl with blue hair",
            "landscape with mountains",
            "cute cat sitting on a chair",
            "cyberpunk city at night",
            "portrait of a warrior"
        ]
        for prompt in safe_prompts:
            result = check_nsfw_prompt(prompt)
            assert result["is_unsafe"] == False, f"Safe prompt blocked: {prompt}"
    
    def test_explicit_content_blocked(self):
        """Explicit sexual content should be blocked"""
        explicit_prompts = [
            "nude woman",
            "naked person",
            "explicit content"
        ]
        for prompt in explicit_prompts:
            result = check_nsfw_prompt(prompt)
            assert result["is_unsafe"] == True, f"Explicit prompt not blocked: {prompt}"
    
    def test_obfuscation_detection(self):
        """Obfuscated explicit terms should be detected"""
        obfuscated = [
            "n1pples",
            "n*ked",
            "s3xual"
        ]
        for prompt in obfuscated:
            result = check_nsfw_prompt(prompt)
            assert result["is_unsafe"] == True, f"Obfuscated prompt not blocked: {prompt}"
    
    def test_artistic_body_terms_allowed(self):
        """Artistic body descriptors should be allowed"""
        artistic_prompts = [
            "slim athletic woman",
            "petite anime character",
            "curvy fashion model",
            "elegant posture portrait"
        ]
        for prompt in artistic_prompts:
            result = check_nsfw_prompt(prompt)
            assert result["is_unsafe"] == False, f"Artistic prompt blocked: {prompt}"
    
    def test_filter_disabled(self):
        """When filter is disabled, nothing should be blocked"""
        result = check_nsfw_prompt("explicit content", nsfw_filter_enabled=False)
        assert result["is_unsafe"] == False
        assert "disabled" in result["message"].lower()
    
    def test_normalize_prompt(self):
        """Test prompt normalization"""
        assert "nipple" in normalize_prompt("n1ppl3")
        assert "sex" in normalize_prompt("s3x")
        assert "naked" in normalize_prompt("n@ked")
    
    def test_empty_prompt(self):
        """Empty prompts should pass"""
        result = check_nsfw_prompt("")
        assert result["is_unsafe"] == False
    
    def test_result_structure(self):
        """Check result dictionary structure"""
        result = check_nsfw_prompt("test prompt")
        assert "is_unsafe" in result
        assert "reason" in result
        assert "category" in result
        assert "message" in result


class TestNormalizePrompt:
    """Test cases for prompt normalization"""
    
    def test_number_substitution(self):
        """Numbers should be converted to letters"""
        assert normalize_prompt("h3ll0") == "hello"
        assert normalize_prompt("t3st") == "test"
    
    def test_special_char_removal(self):
        """Special characters should be removed"""
        assert normalize_prompt("te*st") == "test"
        assert normalize_prompt("te_st") == "test"
        assert normalize_prompt("te-st") == "test"
    
    def test_at_symbol(self):
        """@ should become 'a'"""
        assert normalize_prompt("c@t") == "cat"
    
    def test_lowercase(self):
        """Should convert to lowercase"""
        assert normalize_prompt("TEST") == "test"
