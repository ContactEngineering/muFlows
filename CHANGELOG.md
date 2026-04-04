# Changelog

## v0.1.0

Initial release.

### Features

- **Workflow registry**: `@register_workflow` decorator for registering pure
  workflow functions with optional Pydantic parameter validation
- **Pipeline abstraction**: `Pipeline`, `Step`, and `ForEach` for declarative
  multi-step DAG definitions
- **WorkflowPlan**: Static, serializable DAG representation compiled from
  pipelines via topological sort
- **Content-addressed storage**: Deterministic prefix computation with
  `IdentityKey` annotations for cache control
- **WorkflowContext**: Unified file I/O interface (JSON, xarray, raw bytes)
  with dependency access and progress reporting

### Execution backends

- **LocalBackend**: Synchronous in-process execution for testing and CLI use
- **CeleryBackend**: Parallel DAG execution via Celery chord/group primitives
- **StepFunctionsBackend**: AWS Step Functions orchestration with Lambda

### Storage backends

- **LocalStorageBackend**: Filesystem-based storage with write-once semantics
  and path traversal protection
- **S3StorageBackend**: AWS S3 storage backend

### Utilities

- `run_plan_locally()` helper for pipeline integration testing
- `ResourceManager` and `resolve_uri` for transparent local/remote resource
  fetching
- Extended JSON encoder with NaN, numpy, and datetime support
- xarray Dataset serialization helpers
