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

The `WorkflowContext` class wraps a `StorageBackend` and provides file I/O, dependency access, progress reporting, and workflow parameters. The same context class works with any storage backend (local filesystem or S3).

The `context/` package is split into:

- `context/base.py` — `WorkflowContext` protocol definition (the abstract interface)
- `context/workflow.py` — `WorkflowContext` implementation and `create_local_context()` factory

The `storage/` package provides backend implementations:

- `storage/local.py` — `LocalStorageBackend` (local filesystem)
- `storage/s3.py` — `S3StorageBackend` (AWS S3)

Domain-specific contexts (e.g. `DjangoWorkflowContext` in topobank, `TopographyContext` in sds-workflows) extend `WorkflowContext` or implement the protocol directly.

```python
from muflow import create_local_context
import xarray as xr

# Create a context for local testing
ctx = create_local_context(
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

### Testing Utilities

muFlow provides utilities for testing workflows with automatic dependency resolution:

```python
from muflow import run_plan_locally

# Run a workflow with all dependencies resolved automatically
result = run_plan_locally(
    workflow_name="myapp.training",
    subject_key="dataset:test",
    kwargs={"param": "value"},
    output_dir="/tmp/test_output",
)

# Check if workflow succeeded
assert result.success

# Read output files
data = result.read_json("result.json")
model = result.read_file("model.pkl")

# List all output files
files = result.list_files()
```

The `LocalExecutionResult` provides:
- `success`: Whether the workflow completed without errors
- `plan`: The executed `WorkflowPlan`
- `output_dir`: Path to the root workflow's output directory
- `root_output_dir`: Path to the base output directory
- `error`: Error message if `success` is False
- `read_json(filename)`: Read a JSON file from the output
- `read_file(filename)`: Read raw bytes from the output
- `list_files()`: List all files in the output directory

## Resource Management

muFlow provides utilities for transparent resource fetching from local files or URLs:

```python
from muflow import ResourceManager, is_url, is_local_file, resolve_uri

# Check URI type
is_url("https://example.com/data.nc")  # True
is_url("/local/path/data.nc")  # False
is_local_file("file:///path/data.nc")  # True

# Resolve a single URI (downloads URLs to temp files)
local_path = resolve_uri("https://example.com/data.nc")

# Use ResourceManager for automatic cleanup
with ResourceManager() as rm:
    # Resolve multiple URIs - local files returned as-is,
    # URLs downloaded to temp files
    path1 = rm.resolve("/local/file.nc")
    path2 = rm.resolve("https://example.com/remote.nc")

    # Use the files...

# Temp files automatically cleaned up on context exit
```

Supported URI schemes:
- Local paths: `/path/to/file.nc`, `relative/path.nc`
- File URIs: `file:///path/to/file.nc`
- HTTP/HTTPS: `https://example.com/data.nc`

## License

MIT
