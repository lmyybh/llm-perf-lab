"""Project-level exceptions used outside the CLI presentation layer."""


class LLMPerfError(Exception):
    """Base exception for recoverable llmperf application errors."""


class ValidationError(LLMPerfError):
    """Raised when user-provided input fails semantic validation."""


class ConfigError(LLMPerfError):
    """Raised when runtime configuration is missing or invalid."""


class DatasetFormatError(ValidationError):
    """Raised when a dataset file or record cannot be parsed."""
