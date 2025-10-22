class PyMCToolsError(Exception):
    """Base exception for pymctools."""


class GroupNotFoundError(PyMCToolsError):
    """Raised when a group is not found in InferenceData."""


class CoordinateNotFoundError(PyMCToolsError):
    """Raised when coordinates are not found in InferenceData dimensions."""


class ModelNotFoundError(PyMCToolsError):
    """Raised when model name is not found in InferenceData group."""


class LogLikelihoodNotFoundError(PyMCToolsError):
    """Raised when log-likelihood is not found in InferenceData."""


class NoPosteriorError(PyMCToolsError):
    """Raised when no posterior distribution is found in InferenceData."""


class VariablesNotFoundError(PyMCToolsError):
    """Raised when the desired variable is not found in InferenceData."""
