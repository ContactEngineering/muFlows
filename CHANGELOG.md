# Changelog

## v0.2.0 (not yet released)

### Breaking changes

- Renamed "workflow" to "task" across the entire codebase (`Workflow` -> `Task`, `@register_workflow` -> `@register_task`, etc.).
- Removed `TaskContextProtocol`. Use `TaskContext` directly instead.
- **Caching moved to execution time**: `Pipeline.build_plan()` no longer accepts an `is_cached` callback. Cache detection now happens automatically inside `execute_task()` — if `manifest.json` already exists at the task's storage prefix, the task is skipped without re-running. Remove any `is_cached=...` arguments from `build_plan()` calls and any `LocalStorageBackend.make_cache_checker()` / `is_result_cached()` usage.
- `TaskNode.cached` field removed. Tasks are no longer pre-marked cached at plan-build time.
- `run_plan_locally()` no longer accepts a `use_cache` parameter. Caching is always active and requires no configuration.
- `LocalStorageBackend.make_cache_checker()` and `LocalStorageBackend.is_result_cached()` removed.
- `ExecutionResult` gains a new `cached: bool` field (default `False`) indicating whether the task was skipped due to an existing result.
- `ExecutionResult.files_written` removed. Tasks communicate results exclusively through file writes to the storage context.
- **`submit_plan()` now returns `PlanHandle`** instead of a raw `str`. Update any code that stores or inspects the return value directly.
- `on_node_start`, `on_node_complete`, and `on_node_failure` removed from the `ExecutionBackend` protocol. They remain as keyword-only parameters on `LocalBackend.submit_plan()` for testing and CLI use.
- `CompletionCallback.notify()` signature changed from `(analysis_id: int, result: ExecutionResult)` to `(plan_id: str, success: bool, error: Optional[str])`. Update any custom callback implementations.

### New features

- **`PlanHandle`**: serializable reference to a submitted plan execution. Call `handle.to_json()` to store it (e.g. in a Django model field) and `PlanHandle.from_json(s)` to restore it later. Methods:
  - `get_state() -> str` — `"pending"` | `"running"` | `"success"` | `"failure"`. For Celery hits Redis; for Step Functions calls `describe_execution`; never queries S3.
  - `get_progress() -> PlanProgress` — per-node completion based on `manifest.json` presence.
  - `cancel()` — revokes the Celery chord or stops the Step Functions execution.
  - `configure_celery(app)` — class-level method; call once at startup so `get_state()` / `cancel()` work outside the `CeleryBackend` instance.
- **`PlanProgress`**: returned by `PlanHandle.get_progress()`. Fields: `total`, `completed`, `node_breakdown: dict[str, bool]`. Properties: `fraction` (0.0–1.0), `is_complete`.
- **`ProgressChecker` protocol** (`muflow.storage`): storage-layer abstraction for checking node completion across multiple prefixes without querying an execution backend.
  - `LocalProgressChecker`: checks `os.path.exists(prefix/manifest.json)`.
  - `S3ProgressChecker(bucket)`: one `HEAD` request per prefix (10–50 ms each within the same AWS region).
  - `make_progress_checker(storage_type, storage_config)`: factory that reconstructs the right implementation from serialized config; extend by adding a new class and one branch here.
- **`CeleryCompletionCallback` wired into `CeleryBackend`**: pass `completion_callback=CeleryCompletionCallback(app, "myapp.tasks.on_complete")` to `submit_plan()` and a `muflow.send_completion` Celery task fires after the plan finishes, calling your task with `(plan_id, success, error)`. The `muflow.send_completion` task is registered automatically by `create_celery_task()`.

## v0.1.0 (2026-04-04)

Initial release.

### Features

- **Task registry**: `@register_task` decorator for registering pure
  task functions with optional Pydantic parameter validation
- **Pipeline abstraction**: `Pipeline`, `Step`, and `ForEach` for declarative
  multi-step DAG definitions
- **TaskPlan**: Static, serializable DAG representation compiled from
  pipelines via topological sort
- **Content-addressed storage**: Deterministic prefix computation with
  `IdentityKey` annotations for cache control
- **TaskContext**: Unified file I/O interface (JSON, xarray, raw bytes)
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
