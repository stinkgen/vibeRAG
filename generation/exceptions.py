"""Custom exceptions for the generation module."""

class GenerationError(Exception):
    """Raised when there is an error during text generation."""
    pass

class ProviderNotSupportedError(GenerationError):
    """Raised when the specified provider is not supported."""
    pass

class APIError(GenerationError):
    """Raised when there is an error with the API call."""
    pass 