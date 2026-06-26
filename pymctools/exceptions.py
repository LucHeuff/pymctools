class PyMCToolsError(Exception):
    """Base exception for pymctools."""


class GroupNotFoundError(PyMCToolsError):
    """Raised when a group is not found in xarray.DataTree."""


class CoordinateNotFoundError(PyMCToolsError):
    """Raised when coordinates are not found in xarray.DataTree dimensions."""


class ModelNotFoundError(PyMCToolsError):
    """Raised when model name is not found in xarray.DataTree group."""


class LogLikelihoodNotFoundError(PyMCToolsError):
    """Raised when log-likelihood is not found in xarray.DataTree."""


class NoPosteriorError(PyMCToolsError):
    """Raised when no posterior distribution is found in xarray.DataTree."""


class VariablesNotFoundError(PyMCToolsError):
    """Raised when the desired variable is not found in xarray.DataTree."""
