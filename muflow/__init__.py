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
    Implementations include LambdaBackend and CeleryBackend.

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

# Storage backends
from muflow.storage import (
    StorageBackend,
    LocalStorageBackend,
    S3StorageBackend,
    compute_prefix,
)

# Core context
from muflow.context import (
    WorkflowContext,
    create_local_context,
)

# Plan data structures
from muflow.plan import (
    WorkflowNode,
    WorkflowPlan,
    compute_storage_prefix,
    compute_node_key,
)

# Planner
from muflow.planner import WorkflowPlanner, get_dependency_access_map

# Dependencies
from muflow.dependencies import WorkflowSpec

# Workflow base class
from muflow.workflow import WorkflowImplementation

# Output schema
from muflow.outputs import OutputFile, get_outputs_schema

# Registry
from muflow import registry
from muflow.registry import WorkflowEntry, register_workflow

# Executor
from muflow.executor import (
    ExecutionPayload,
    ExecutionResult,
    execute_workflow,
)

# Execution backends
from muflow.backends import ExecutionBackend
from muflow.backends.base import LocalBackend

# Lambda backend factory (requires boto3)
try:
    from muflow.backends.lambda_backend import create_lambda_handler
except ImportError:
    pass

# I/O utilities
from muflow.io import (
    ExtendedJSONEncoder,
    dumps_json,
    loads_json,
    load_xarray_from_bytes,
    save_xarray_to_bytes,
)

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
    # Workflow
    "WorkflowImplementation",
    # Outputs
    "OutputFile",
    "get_outputs_schema",
    # Registry
    "registry",
    "WorkflowEntry",
    "register_workflow",
    # Executor
    "ExecutionPayload",
    "ExecutionResult",
    "execute_workflow",
    # Plan
    "WorkflowNode",
    "WorkflowPlan",
    "compute_storage_prefix",
    "compute_node_key",
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
]

# Optional: LambdaBackend (requires boto3)
try:
    from muflow.backends import LambdaBackend
    __all__.extend(["LambdaBackend", "create_lambda_handler"])
except ImportError:
    pass

# Optional: CeleryBackend (requires celery)
try:
    from muflow.backends import CeleryBackend, create_celery_task
    from muflow.backends.callbacks import (
        CompletionCallback,
        CeleryCompletionCallback,
        NoOpCompletionCallback,
    )
    __all__.extend([
        "CeleryBackend",
        "create_celery_task",
        "CompletionCallback",
        "CeleryCompletionCallback",
        "NoOpCompletionCallback",
    ])
except ImportError:
    pass
