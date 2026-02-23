"""Unit tests for validation utilities."""

import pytest
from price_intelligence.exceptions import ValidationError
from price_intelligence.utils import validate_string_input, validate_numeric_input


class TestStringValidation:
    """Tests for string input validation."""
    
    def test_valid_string(self):
        """Test valid string input."""
        result = validate_string_input("hello world")
        assert result == "hello world"
    
    def test_string_with_whitespace(self):
        """Test string with leading/trailing whitespace is stripped."""
        result = validate_string_input("  hello world  ")
        assert result == "hello world"
    
    def test_empty_string(self):
        """Test empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_string_input("")
    
    def test_string_too_long(self):
        """Test string exceeding max_length raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_string_input("x" * 1001, max_length=1000)
    
    def test_non_string_input(self):
        """Test non-string input raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_string_input(123)


class TestNumericValidation:
    """Tests for numeric input validation."""
    
    def test_valid_integer(self):
        """Test valid integer input."""
        result = validate_numeric_input(42)
        assert result == 42.0
    
    def test_valid_float(self):
        """Test valid float input."""
        result = validate_numeric_input(3.14)
        assert result == 3.14
    
    def test_string_number(self):
        """Test string representation of number."""
        result = validate_numeric_input("42")
        assert result == 42.0
    
    def test_below_minimum(self):
        """Test value below minimum raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_numeric_input(5, min_val=10)
    
    def test_above_maximum(self):
        """Test value above maximum raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_numeric_input(15, max_val=10)
    
    def test_non_numeric_input(self):
        """Test non-numeric input raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_numeric_input("not_a_number")
