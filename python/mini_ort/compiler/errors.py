"""Compiler-specific errors."""


class ModelFormatError(ValueError):
    """Raised when a .mer artifact violates its binary or model contract."""
