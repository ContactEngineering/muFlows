# muFlows

Backend-agnostic workflow execution engine.

## Overview

muFlows provides abstractions for defining and executing workflows that can run on multiple backends (Celery, AWS Lambda, AWS Batch) without modification.

## Installation

```bash
pip install muflows

# With S3 support (for Lambda/Batch)
pip install muflows[s3]

# For development
pip install muflows[dev]
```

## Core Concepts

### WorkflowContext

Abstract interface (Protocol) for file I/O that workflow implementations use. The context is passed to `WorkflowImplementation.execute(context)` and provides storage, dependency access, and progress reporting. This allows the same workflow code to run on different backends without modification.

The `context/` package is split into separate modules:

- `context/base.py` — `WorkflowContext` protocol definition (the abstract interface)
- `context/local.py` — `LocalFolderContext` (local filesystem, for testing and local execution)
- `context/s3.py` — `S3WorkflowContext` (direct S3 access, for Lambda/Batch)

Domain-specific contexts (e.g. `DjangoWorkflowContext` in topobank, `TopographyContext` in sds-workflows) extend `LocalFolderContext` or implement the protocol directly, adding domain properties like `topography` or `topography_name`.

```python
from muflow import LocalFolderContext
import xarray as xr

# Create a context
ctx = LocalFolderContext(
    path="/tmp/workflow-output",
    kwargs={"param1": "value1"},
)

# Use the context for I/O
ctx.save_json("result.json", {"accuracy": 0.95})
ctx.save_xarray("model.nc", xr.Dataset({"weights": [1, 2, 3]}))

# Read back
result = ctx.read_json("result.json")
model = ctx.read_xarray("model.nc")
```

### WorkflowPlan

A static DAG representing the complete execution plan. Plans are computed once upfront and stored as JSON.

```python
from muflows import WorkflowPlan, WorkflowNode

# Create nodes
nodes = {
    "preprocess": WorkflowNode(
        key="preprocess",
        function="my.preprocess",
        subject_key="data:123",
        kwargs={},
        storage_prefix="results/preprocess/abc123",
    ),
    "train": WorkflowNode(
        key="train",
        function="my.train",
        subject_key="data:123",
        kwargs={"epochs": 10},
        storage_prefix="results/train/def456",
        depends_on=["preprocess"],
    ),
}

# Create plan
plan = WorkflowPlan(nodes=nodes, root_key="train")

# Serialize to JSON
json_str = plan.to_json()

# Find ready nodes
ready = plan.ready_nodes(completed={"preprocess"})
```

### ExecutionBackend

Interface for dispatching workflow nodes to compute backends:

- `LocalBackend` - Synchronous execution (for testing)
- `LambdaBackend` - AWS Lambda
- `CeleryBackend` - Celery (in topobank, not here)

```python
from muflows import LambdaBackend

backend = LambdaBackend(
    function_name="my-workflow-function",
    bucket="my-bucket",
)

task_id = backend.submit(analysis_id=123, payload={
    "function": "my.workflow",
    "kwargs": {"param": "value"},
    "storage_prefix": "results/abc123",
})
```

## Content-Addressed Storage

muFlows uses deterministic, content-addressed storage prefixes:

```python
from muflows import compute_storage_prefix

prefix = compute_storage_prefix(
    function_name="my.workflow",
    subject_key="data:123",
    kwargs={"param": "value"},
)
# Returns: "data-lake/results/my.workflow/a1b2c3d4..."
```

Same inputs always produce the same prefix, enabling automatic caching.

## Testing

```bash
pip install muflows[test]
pytest
```

## License

MIT
