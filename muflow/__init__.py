"""muflow - Backend-agnostic workflow execution engine.

muflow provides abstractions for defining and executing workflows that can
run on multiple backends (Celery, AWS Lambda, AWS Batch) without modification.

Core Concepts
-------------
WorkflowContext
    Unified context class for workflow file I/O, dependency access, and
    progress reporting. Works with any StorageBackend (LocalStorageBackend,
    S3StorageBackend).

StorageBackend
    Abstract interface for file storage. Implementations include
    LocalStorageBackend (filesystem) and S3StorageBackend (AWS S3).

WorkflowPlan
    A static DAG representing the complete execution plan. Built once upfront
    by the WorkflowPlanner and stored as JSON.

WorkflowPlanner
    Builds execution DAGs from workflow dependency declarations.

ExecutionBackend
    Interface for dispatching workflow nodes to different compute backends.
    Implementations include StepFunctionsBackend and CeleryBackend.

Example
-------
>>> from muflow import create_local_context
>>> import xarray as xr
>>>
>>> # Create a context for testing
>>> ctx = create_local_context(
...     path="/tmp/workflow-output",
...     kwargs={"param1": "value1"},
... )
>>>
>>> # Use the context for I/O
>>> ctx.save_json("result.json", {"accuracy": 0.95})
>>> ctx.save_xarray("model.nc", xr.Dataset({"weights": [1, 2, 3]}))
>>>
>>> # Read back
>>> result = ctx.read_json("result.json")
>>> model = ctx.read_xarray("model.nc")
"""

__version__ = "0.1.0"

# Registry
from muflow import registry

# Execution backends
from muflow.backends import ExecutionBackend, LocalBackend

# Core context
from muflow.context import WorkflowContext, create_local_context

# Dependencies
from muflow.dependencies import WorkflowSpec

# Executor
from muflow.executor import ExecutionPayload, ExecutionResult, execute_workflow

# Output schema
from muflow.outputs import OutputFile, get_outputs_schema

# Plan data structures
from muflow.plan import WorkflowNode, WorkflowPlan

# Planner
from muflow.planner import WorkflowPlanner, get_dependency_access_map
from muflow.registry import IdentityKey, WorkflowEntry, register_workflow

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
    "WorkflowContext",
    "create_local_context",
    # Outputs
    "OutputFile",
    "get_outputs_schema",
    # Registry
    "registry",
    "IdentityKey",
    "WorkflowEntry",
    "register_workflow",
    # Executor
    "ExecutionPayload",
    "ExecutionResult",
    "execute_workflow",
    # Plan
    "WorkflowNode",
    "WorkflowPlan",
    # Planner
    "WorkflowPlanner",
    "get_dependency_access_map",
    # Dependencies
    "WorkflowSpec",
    # Backends
    "ExecutionBackend",
    "LocalBackend",
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
    from muflow.backends import StepFunctionsBackend, create_lambda_handler  # noqa: F401
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
