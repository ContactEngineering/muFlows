"""muflow - Backend-agnostic task execution engine.

muflow provides abstractions for defining and executing tasks that can
run on multiple backends (Celery, AWS Lambda, AWS Step Functions) without
modification.

Tasks are registered as pure computational units via
``@register_task``.  DAG topology is declared separately via
:class:`Pipeline`, which compiles to a :class:`TaskPlan` that any
backend can execute.

Example
-------
>>> from muflow import register_task, Pipeline, Step, ForEach
>>>
>>> @register_task(name="ml.train")
... def train(context):
...     context.save_json("model.json", {"weights": []})
>>>
>>> pipeline = Pipeline(
...     name="ml.pipeline",
...     steps={"train": Step(task="ml.train")},
... )
>>> plan = pipeline.build_plan("tag:1", {})
"""

__version__ = "0.1.0"

# Registry
from muflow import registry

# Execution backends
from muflow.backends import ExecutionBackend, LocalBackend, PlanHandle
from muflow.backends.handle import PlanProgress

# Core context
from muflow.context import TaskContext, create_local_context

# Executor
from muflow.executor import ExecutionPayload, ExecutionResult, execute_task

# Output schema
from muflow.outputs import OutputFile, get_outputs_schema

# Plan data structures
from muflow.plan import TaskNode, TaskPlan

# Pipeline
from muflow.pipeline import ForEach, Pipeline, Step

# Registry
from muflow.registry import IdentityKey, TaskEntry, register_task

# Storage backends
from muflow.storage import (
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
    compute_prefix,
)

# I/O utilities
from muflow.io import (
    ExtendedJSONEncoder,
    ResourceManager,
    dumps_json,
    is_local_file,
    is_url,
    load_xarray_from_bytes,
    loads_json,
    resolve_uri,
    save_xarray_to_bytes,
)

# Testing utilities
from muflow.testing import LocalExecutionResult, run_plan_locally

__all__ = [
    # Version
    "__version__",
    # Storage
    "StorageBackend",
    "LocalStorageBackend",
    "S3StorageBackend",
    "compute_prefix",
    # Context
    "TaskContext",
    "create_local_context",
    # Outputs
    "OutputFile",
    "get_outputs_schema",
    # Registry
    "registry",
    "IdentityKey",
    "TaskEntry",
    "register_task",
    # Executor
    "ExecutionPayload",
    "ExecutionResult",
    "execute_task",
    # Plan
    "TaskNode",
    "TaskPlan",
    # Pipeline
    "Pipeline",
    "Step",
    "ForEach",
    # Backends
    "ExecutionBackend",
    "LocalBackend",
    "PlanHandle",
    "PlanProgress",
    # I/O
    "ExtendedJSONEncoder",
    "dumps_json",
    "loads_json",
    "load_xarray_from_bytes",
    "save_xarray_to_bytes",
    # Resource fetching
    "is_url",
    "is_local_file",
    "resolve_uri",
    "ResourceManager",
    # Testing
    "LocalExecutionResult",
    "run_plan_locally",
]

# Optional: StepFunctionsBackend (requires boto3)
try:
    from muflow.backends import (  # noqa: F401
        StepFunctionsBackend, create_lambda_handler,
    )
    __all__.extend(["StepFunctionsBackend", "create_lambda_handler"])
except ImportError:
    pass

# Completion callbacks (always available)
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

# Optional: CeleryBackend (requires celery)
try:
    from muflow.backends import CeleryBackend, create_celery_task  # noqa: F401
    __all__.extend(["CeleryBackend", "create_celery_task"])
except ImportError:
    pass
