"""Unit tests for parsing utilities."""

import pytest
from price_intelligence.exceptions import PriceParsingError
from price_intelligence.utils import extract_price, extract_text_content


class TestPriceParsing:
    """Tests for price extraction."""
    
    def test_extract_simple_price(self):
        """Test extracting simple price."""
        text = "Price is $49.99 and that's final"
        price = extract_price(text)
        assert price == 49.99
    
    def test_extract_integer_price(self):
        """Test extracting integer price."""
        text = "Price is $100"
        price = extract_price(text)
        assert price == 100.0
    
    def test_extract_price_with_comma(self):
        """Test extracting price with comma."""
        text = "Price is $1,299.99"
        price = extract_price(text)
        assert price == 1299.99
    
    def test_missing_prefix(self):
        """Test missing prefix raises error."""
        text = "Cost is $49.99"
        with pytest.raises(PriceParsingError):
            extract_price(text, prefix="Price is $")
    
    def test_no_numeric_value(self):
        """Test text without numeric value raises error."""
        text = "Price is $ none"
        with pytest.raises(PriceParsingError):
            extract_price(text)
    
    def test_negative_price_rejected(self):
        """Test negative price is rejected."""
        text = "Price is $-100"
        with pytest.raises(PriceParsingError):
            extract_price(text)
    
    def test_unreasonably_large_price_rejected(self):
        """Test unreasonably large price is rejected."""
        text = "Price is $10000000"
        with pytest.raises(PriceParsingError):
            extract_price(text)


class TestTextContentExtraction:
    """Tests for text content extraction."""
    
    def test_extract_with_start_marker(self):
        """Test extraction with start marker."""
        text = "PREFIX content SUFFIX"
        result = extract_text_content(text, start_marker="PREFIX ")
        assert result == "content SUFFIX"
    
    def test_extract_with_end_marker(self):
        """Test extraction with end marker."""
        text = "PREFIX content SUFFIX"
        result = extract_text_content(text, end_marker=" SUFFIX")
        assert result == "PREFIX content"
    
    def test_extract_with_both_markers(self):
        """Test extraction with both markers."""
        text = "PREFIX content SUFFIX"
        result = extract_text_content(text, start_marker="PREFIX ", end_marker=" SUFFIX")
        assert result == "content"
    
    def test_extract_missing_markers(self):
        """Test extraction when markers not found."""
        text = "content"
        result = extract_text_content(text, start_marker="PREFIX")
        assert result == "content"
