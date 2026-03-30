"""Execution backends for muflow.

Backends receive an entire WorkflowPlan and orchestrate execution using
their native primitives:

- LocalBackend: Serial execution in-process (for testing/CLI)
- CeleryBackend: Parallel execution via Celery chord/group
- LambdaBackend: AWS Lambda execution (serial, or Step Functions for parallel)
"""

from muflow.backends.base import ExecutionBackend, LocalBackend

__all__ = ["ExecutionBackend", "LocalBackend"]

# LambdaBackend is optional (requires boto3)
try:
    from muflow.backends.aws_lambda import (  # noqa: F401
        LambdaBackend,
        create_lambda_handler,
    )
    __all__.extend(["LambdaBackend", "create_lambda_handler"])
except ImportError:
    pass

# CeleryBackend is optional (requires celery)
try:
    from muflow.backends.celery import CeleryBackend, create_celery_task  # noqa: F401
    __all__.extend(["CeleryBackend", "create_celery_task"])
except ImportError:
    pass

# Callbacks (Celery optional)
try:
    from muflow.backends.callbacks import (  # noqa: F401
        CeleryCompletionCallback,
        CompletionCallback,
        LoggingCompletionCallback,
        NoOpCompletionCallback,
    )
    __all__.extend([
        "CompletionCallback",
        "CeleryCompletionCallback",
        "NoOpCompletionCallback",
        "LoggingCompletionCallback",
    ])
except ImportError:
    pass
