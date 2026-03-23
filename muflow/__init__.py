"""muflow - Backend-agnostic workflow execution engine.

muflow provides abstractions for defining and executing workflows that can
run on multiple backends (Celery, AWS Lambda, AWS Batch) without modification.

Core Concepts
-------------
WorkflowContext
    Abstract interface for file I/O that workflow implementations use.
    Implementations include S3WorkflowContext (for Lambda/Batch) and
    LocalFolderContext (for testing). Django integration (DjangoWorkflowContext)
    lives in topobank, not here.

WorkflowPlan
    A static DAG representing the complete execution plan. Built once upfront
    by the WorkflowPlanner (in topobank) and stored as JSON.

ExecutionBackend
    Interface for dispatching workflow nodes to different compute backends.
    Implementations include LambdaBackend (here) and CeleryBackend (in topobank).

Example
-------
>>> from muflow import LocalFolderContext
>>> import xarray as xr
>>>
>>> # Create a context for testing
>>> ctx = LocalFolderContext(
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

# Core context abstractions
from muflow.context import (
    WorkflowContext,
    LocalFolderContext,
    S3WorkflowContext,
)

# Plan data structures
from muflow.plan import (
    WorkflowNode,
    WorkflowPlan,
    compute_storage_prefix,
    compute_node_key,
)

# Workflow base class
from muflow.workflow import WorkflowImplementation

# Output schema
from muflow.outputs import OutputFile, get_outputs_schema

# Registry
from muflow import registry

# Executor
from muflow.executor import (
    ExecutionPayload,
    ExecutionResult,
    execute_workflow,
)

# Execution backends
from muflow.backends import ExecutionBackend
from muflow.backends.base import LocalBackend

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
    # Context
    "WorkflowContext",
    "LocalFolderContext",
    "S3WorkflowContext",
    # Workflow
    "WorkflowImplementation",
    # Outputs
    "OutputFile",
    "get_outputs_schema",
    # Registry
    "registry",
    # Executor
    "ExecutionPayload",
    "ExecutionResult",
    "execute_workflow",
    # Plan
    "WorkflowNode",
    "WorkflowPlan",
    "compute_storage_prefix",
    "compute_node_key",
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
    __all__.append("LambdaBackend")
except ImportError:
    pass
