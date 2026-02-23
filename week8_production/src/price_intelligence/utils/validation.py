"""Input validation utilities."""

from typing import Any
from ..exceptions import ValidationError


def validate_string_input(value: Any, min_length: int = 1, max_length: int = 10000, 
                         field_name: str = "input") -> str:
    """
    Validate and return a string input.
    
    Args:
        value: Value to validate
        min_length: Minimum string length
        max_length: Maximum string length
        field_name: Name of field for error messages
        
    Returns:
        str: Validated string value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    
    if len(value) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters")
    
    if len(value) > max_length:
        raise ValidationError(f"{field_name} must be at most {max_length} characters")
    
    return value.strip()


def validate_numeric_input(value: Any, min_val: float = None, max_val: float = None,
                          field_name: str = "value") -> float:
    """
    Validate and return a numeric input.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error messages
        
    Returns:
        float: Validated numeric value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be numeric, got {type(value).__name__}")
    
    if min_val is not None and numeric_value < min_val:
        raise ValidationError(f"{field_name} must be >= {min_val}")
    
    if max_val is not None and numeric_value > max_val:
        raise ValidationError(f"{field_name} must be <= {max_val}")
    
    return numeric_value
