"""muflows - Backend-agnostic workflow execution engine.

muflows provides abstractions for defining and executing workflows that can
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
>>> from muflows import LocalFolderContext
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
from muflows.context import (
    WorkflowContext,
    LocalFolderContext,
    S3WorkflowContext,
)

# Plan data structures
from muflows.plan import (
    WorkflowNode,
    WorkflowPlan,
    compute_storage_prefix,
    compute_node_key,
)

# Workflow base class
from muflows.workflow import WorkflowImplementation

# Output schema
from muflows.outputs import OutputFile, get_outputs_schema

# Registry
from muflows import registry

# Executor
from muflows.executor import (
    ExecutionPayload,
    ExecutionResult,
    execute_workflow,
)

# Execution backends
from muflows.backends import ExecutionBackend
from muflows.backends.base import LocalBackend

# I/O utilities
from muflows.io import (
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
    from muflows.backends import LambdaBackend
    __all__.append("LambdaBackend")
except ImportError:
    pass
