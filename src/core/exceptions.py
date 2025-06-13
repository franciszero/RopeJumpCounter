"""
Exception definitions module

Defines custom exception classes used throughout the RopeJumpCounter application.
These exceptions provide specific error handling for different components.
"""


class AppError(Exception):
    """Base application exception

    All custom exceptions in the application inherit from this base class.
    This allows for centralized exception handling and logging.
    """
    pass


class CameraError(AppError):
    """Camera-related errors

    Raised when camera initialization, capture, or processing fails.
    Common scenarios include device not found, permission denied, or format issues.
    """
    pass


class ModelError(AppError):
    """Machine learning model errors

    Raised when model loading, inference, or validation fails.
    This includes file not found, incompatible formats, or prediction errors.
    """
    pass


class ConfigError(AppError):
    """Configuration-related errors

    Raised when configuration loading, parsing, or validation fails.
    This includes missing files, invalid values, or schema mismatches.
    """
    pass
