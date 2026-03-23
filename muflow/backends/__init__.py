"""Execution backends for muflow."""

from muflow.backends.base import ExecutionBackend

__all__ = ["ExecutionBackend"]

# LambdaBackend is optional (requires boto3)
try:
    from muflow.backends.lambda_backend import LambdaBackend
    __all__.append("LambdaBackend")
except ImportError:
    pass
