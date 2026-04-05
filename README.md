# muFlow

*Pronounced "microflow" (μFlow).*

Backend-agnostic task execution engine.

## Overview

muFlow provides abstractions for defining and executing tasks that can run on multiple backends (Celery, AWS Lambda, AWS Step Functions) without modification.

Tasks are registered as **pure computational units** via `@register_task`. DAG topology is declared separately in a **Pipeline** definition, keeping task logic decoupled from orchestration.

## Installation

```bash
pip install muflow

# With S3 support (for Lambda/Step Functions)
pip install muflow[s3]

# For development
pip install muflow[dev]
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    User Code                    │
│  @register_task()    Pipeline(steps={...})  │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Pipeline.build_plan │
              │   (compiles steps    │
              │    into static DAG)  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   TaskPlan (DAG) │
              │   nodes, root_key    │
              └──────────┬───────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      ┌────────┐  ┌─────────┐  ┌──────────────┐
      │ Local  │  │ Celery  │  │ Step         │
      │Backend │  │ Backend │  │ Functions    │
      └────────┘  └─────────┘  └──────────────┘
```

## Core Concepts

### Registering Tasks

Tasks are registered as plain functions:

```python
from muflow import register_task

@register_task(name="ml.compute_features")
def compute_features(context):
    ds = context.kwargs["dataset_name"]
    # ... compute features ...
    context.save_json("features.json", {"dataset": ds, "features": [...]})
```

### TaskContext

The `TaskContext` wraps a `StorageBackend` and provides file I/O, dependency access, and progress reporting:

```python
from muflow import create_local_context

ctx = create_local_context(
    path="/tmp/task-output",
    kwargs={"param1": "value1"},
)

# Write outputs
ctx.save_json("result.json", {"accuracy": 0.95})
ctx.save_xarray("model.nc", xr.Dataset({"weights": [1, 2, 3]}))

# Read back
result = ctx.read_json("result.json")

# Access upstream dependencies
for key in ctx.dependency_keys():
    dep = ctx.dependency(key)
    data = dep.read_json("features.json")
```

### Pipelines (Recommended for Multi-Step Tasks)

The `Pipeline` abstraction lets you declare the full DAG in one place. Individual tasks remain pure — they have no knowledge of the DAG topology.

```python
from muflow import Pipeline, Step, ForEach

ml_pipeline = Pipeline(
    name="ml.full_pipeline",
    display_name="ML Training Pipeline",
    steps={
        "features": ForEach(
            task="ml.compute_features",
            over=lambda subject_key, kwargs: [
                {"dataset_name": ds} for ds in kwargs["datasets"]
            ],
        ),
        "train": Step(
            task="ml.train_model",
            after=["features"],
        ),
        "loo_cv": ForEach(
            task="ml.loo_cv",
            after=["features"],
            over=lambda subject_key, kwargs: [
                {"leave_out_index": i, "datasets": kwargs["datasets"]}
                for i in range(len(kwargs["datasets"]))
            ],
        ),
        "reports": ForEach(
            task="ml.generate_report",
            after=["train", "loo_cv"],
            over=lambda subject_key, kwargs: [
                {"format": fmt} for fmt in ("pdf", "xlsx", "csv")
            ],
        ),
    },
)

# Build the execution plan
plan = ml_pipeline.build_plan(
    subject_key="experiment:1",
    kwargs={"datasets": ["dataset_a", "dataset_b", "dataset_c"]},
)

# Execute on any backend
backend.submit_plan(plan)
```

This produces the following DAG (for N=3 datasets, 11 nodes total):

```
[features:0]  [features:1]  [features:2]     ← 3 parallel leaf nodes
      │              │              │
[train_model] [loo_cv:0] [loo_cv:1] [loo_cv:2]  ← 1 + 3 parallel
      │           │          │          │
[report_pdf] [report_xlsx] [report_csv]  ← 3 parallel
      │            │             │
           [sentinel/root]               ← auto-created
```

#### Step Types

- **`Step(task, after)`** — a single job. Use `kwargs_map` to compute step-specific kwargs.
- **`ForEach(task, over, after)`** — fan-out: `over(subject_key, kwargs)` returns a list of per-job kwargs dicts. One node is created per item.

#### Dependency Access Keys

When a downstream step references an upstream `ForEach` step, the access keys use colon-indexed notation:

```python
# upstream "features" has 3 jobs → access keys are "features:0", "features:1", "features:2"
# upstream "train" has 1 job → access key is just "train"

@register_task(name="ml.generate_report")
def generate_report(context):
    model = context.dependency("train").read_json("model.json")
    for key in context.dependency_keys():
        if key.startswith("loo_cv:"):
            cv = context.dependency(key).read_json("cv_result.json")
```

### TaskPlan

A static DAG representing the complete execution plan. Plans are compiled from a Pipeline definition once upfront and can be serialized as JSON.

```python
from muflow import TaskPlan

# Build from a pipeline
plan = my_pipeline.build_plan("tag:1", {"param": "value"})

# Inspect the plan
print(f"Total nodes: {len(plan.nodes)}")
print(f"Leaf nodes: {[n.function for n in plan.leaf_nodes()]}")

# Walk through execution order
completed = set()
while not plan.is_complete(completed):
    ready = plan.ready_nodes(completed)
    for node in ready:
        execute(node)
        completed.add(node.key)

# Serialize to JSON
json_str = plan.to_json()
```

### Content-Addressed Storage

muFlow uses deterministic, content-addressed storage prefixes. Same inputs always produce the same prefix, enabling automatic caching:

```python
from muflow import compute_prefix

prefix = compute_prefix(
    {"task": "my.task", "subject": "data:123", "param": "value"},
)
# Returns: "muflow/my.task/a1b2c3d4..."
```

### Caching

Caching is automatic and requires no configuration. When `execute_task()` runs a node, it first checks whether `manifest.json` already exists at the node's storage prefix. If it does, the task is skipped and marked cached — the node counts as completed and downstream tasks proceed normally.

```python
# Re-running the same plan reuses any already-complete nodes automatically.
plan = ml_pipeline.build_plan(
    subject_key="experiment:1",
    kwargs={"datasets": ["a", "b", "c"]},
)
backend.submit_plan(plan)
# feature nodes with existing manifest.json are skipped instantly
```

### Identity Keys

Use `IdentityKey` annotations to control which parameters affect caching. Only identity-keyed fields are included in the content-addressed hash:

```python
from typing import Annotated
import pydantic
from muflow import IdentityKey, register_task

class TrainParams(pydantic.BaseModel):
    dataset_id: Annotated[int, IdentityKey()]  # affects hash
    display_name: str                           # does not affect hash

@register_task(name="ml.train", parameters=TrainParams)
def train(context):
    # context.kwargs is a validated TrainParams instance
    print(context.kwargs.dataset_id)
```

### Execution Backends

- **`LocalBackend`** — synchronous in-process execution (for testing)
- **`CeleryBackend`** — Celery chord/group for parallel execution
- **`StepFunctionsBackend`** — AWS Step Functions with Lambda

`submit_plan()` returns a `PlanHandle` — a serialisable reference to the submitted execution:

```python
from muflow.backends import LocalBackend

backend = LocalBackend(base_path="/tmp/results")
handle = backend.submit_plan(plan)

# Query state (no S3 queries; uses Redis for Celery, sfn.describe_execution for SFN)
print(handle.get_state())   # "success" | "failure" | "running" | "pending"

# Check per-node progress (checks manifest.json at each node's storage prefix)
progress = handle.get_progress()
print(f"{progress.completed}/{progress.total} nodes complete")

# Persist across processes (e.g. store in a Django model field)
stored = handle.to_json()
handle = PlanHandle.from_json(stored)
```

## Testing

```bash
pip install muflow[dev]
pytest
```

### Testing Utilities

muFlow provides utilities for testing pipelines locally:

```python
from muflow import run_plan_locally

result = run_plan_locally(
    pipeline=my_pipeline,
    subject_key="dataset:test",
    kwargs={"param": "value"},
    output_dir="/tmp/test_output",
)

assert result.success
data = result.read_json("result.json")
files = result.list_files()
```

## Resource Management

Utilities for transparent resource fetching from local files or URLs:

```python
from muflow import ResourceManager, is_url, resolve_uri

# Resolve a URI (downloads URLs to temp files)
local_path = resolve_uri("https://example.com/data.nc")

# Automatic cleanup with context manager
with ResourceManager() as rm:
    path1 = rm.resolve("/local/file.nc")
    path2 = rm.resolve("https://example.com/remote.nc")
```

## License

MIT
