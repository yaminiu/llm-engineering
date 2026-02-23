"""Custom exceptions for Price Intelligence service."""


class PriceIntelligenceError(Exception):
    """Base exception for all Price Intelligence errors."""

    pass


class ModelLoadError(PriceIntelligenceError):
    """Raised when model loading fails."""

    pass


class ModelInferenceError(PriceIntelligenceError):
    """Raised when model inference fails."""

    pass


class PriceParsingError(PriceIntelligenceError):
    """Raised when price parsing fails."""

    pass


class ValidationError(PriceIntelligenceError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(PriceIntelligenceError):
    """Raised when configuration is invalid."""

    pass
