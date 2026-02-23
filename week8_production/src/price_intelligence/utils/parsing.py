"""Parsing utilities for extracting values from model outputs."""

import re
from typing import Optional
from ..exceptions import PriceParsingError


def extract_price(text: str, prefix: str = "Price is $") -> float:
    """
    Extract price value from text output.
    
    Args:
        text: Text containing price information
        prefix: Expected prefix before price
        
    Returns:
        float: Extracted price value
        
    Raises:
        PriceParsingError: If price cannot be extracted
    """
    try:
        # Find the section after the prefix
        if prefix not in text:
            raise PriceParsingError(f"Expected prefix '{prefix}' not found in output")
        
        contents = text.split(prefix)[1]
        contents = contents.replace(",", "").strip()
        
        # Extract numeric value using regex
        match = re.search(r"[-+]?\d*\.?\d+", contents)
        
        if not match:
            raise PriceParsingError(f"No numeric value found after '{prefix}' in '{text}'")
        
        price = float(match.group())
        
        # Validate price is reasonable (positive, not absurdly large)
        if price < 0:
            raise PriceParsingError(f"Negative price extracted: ${price}")
        
        if price > 1_000_000:
            raise PriceParsingError(f"Unreasonably large price extracted: ${price}")
        
        return price
        
    except PriceParsingError:
        raise
    except Exception as e:
        raise PriceParsingError(f"Failed to parse price from '{text}': {str(e)}")


def extract_text_content(text: str, start_marker: Optional[str] = None, 
                        end_marker: Optional[str] = None) -> str:
    """
    Extract text content between markers.
    
    Args:
        text: Text to extract from
        start_marker: Start marker (inclusive)
        end_marker: End marker (exclusive)
        
    Returns:
        str: Extracted content
    """
    content = text
    
    if start_marker and start_marker in content:
        content = content.split(start_marker)[1]
    
    if end_marker and end_marker in content:
        content = content.split(end_marker)[0]
    
    return content.strip()
