# muFlow Design Notes

This document explains the architectural decisions behind muFlow and the rationale for how it is structured. It is intended for contributors and integrators who need to understand *why* things are the way they are, not just *what* they do.

---

## Separation of concerns

muFlow separates four concerns that are often entangled in task-execution frameworks:

| Concern | Where it lives |
|---------|----------------|
| **Task logic** | `@register_task` functions — pure functions that read inputs and write outputs |
| **DAG topology** | `Pipeline` / `Step` / `ForEach` — declarative, no task code |
| **Orchestration** | `ExecutionBackend` (Local / Celery / Step Functions) — drives execution |
| **Storage I/O** | `StorageBackend` (Local / S3) — abstracts file reads and writes |

The key consequence: a task function is never aware of which backend it runs on, which other tasks exist in the pipeline, or what storage system is in use. The `TaskContext` it receives is always the same interface regardless.

---

## Content-addressed storage

Every task execution maps to a deterministic storage prefix:

```
{base_prefix}/{task_name}/{sha256_hex[:16]}
```

The hash is computed over a sorted JSON serialisation of the task's *identity dict* — by default all kwargs, but controllable via `IdentityKey` annotations. Fields not annotated with `IdentityKey` are excluded from the hash, allowing non-identity fields (display names, description strings) to change without invalidating cached results.

```python
class TrainParams(pydantic.BaseModel):
    dataset_id: Annotated[int, IdentityKey()]  # in hash
    display_name: str                           # not in hash
```

Because the prefix is deterministic, the same task with the same inputs always maps to the same directory or S3 prefix. This is the foundation for caching.

---

## Caching

### Design decision: cache detection at execution time, not plan-build time

Early versions detected cached nodes during `Pipeline.build_plan()` via an `is_cached` callback, pre-marking `TaskNode.cached = True`. This was removed. Cache detection now happens inside `execute_task()` at the start of each node's execution:

```python
def execute_task(payload, context, get_entry) -> ExecutionResult:
    if context.storage.is_cached():           # manifest.json present?
        return ExecutionResult(success=True, cached=True)
    try:
        ...
    finally:
        context.storage.write_manifest()      # always write on completion
```

**Why:** The async backends (Celery, Step Functions) execute nodes in separate processes or Lambda functions. Those processes have no shared state with the process that built the plan — they only receive a serialised `ExecutionPayload`. Moving the cache check into the executor means it works identically on every backend without any plumbing changes. The `manifest.json` is the single source of truth for whether a node is complete.

### What `manifest.json` records

```json
{
  "files": ["features.json", "model.nc"],
  "timestamp": "2026-04-05T12:34:56+00:00"
}
```

Its *presence* is the cache signal. Its *contents* are metadata. A task is considered complete — and will be skipped on re-execution — if and only if `manifest.json` exists at its storage prefix.

The `finally` block in `execute_task()` calls `write_manifest()` unconditionally — it runs whether the task succeeded or raised an exception. A failed task therefore also writes `manifest.json`, listing whatever files were written before the failure. Because `is_cached()` only checks for presence, that node will be treated as cached on any subsequent execution and silently skipped.

**Consequence:** a failed node cannot be re-executed automatically. To re-run it, the `manifest.json` (or the entire storage prefix) must be deleted first.

---

## Tasks communicate through files, not return values

Task functions write results to `context.save_*()` and downstream tasks read them via `context.dependency(key).read_*()`. There are no return values between tasks.

This is enforced structurally:
- Celery tasks use `immutable=True` signatures — the chord result is discarded and never injected as an argument.
- Step Functions uses `ResultPath: null` in the ASL — Lambda return values are discarded by the state machine.
- `ExecutionResult` (the internal result object) does not carry file contents — only `success`, `cached`, and error information.

**Why:** Return values would require every node's output to pass through the broker (Redis for Celery, Step Functions state for SFN), creating size limits and coupling the data path to the orchestration path. S3 / local filesystem are the right data path; the orchestration layer only needs to know success or failure.

---

## `PlanHandle`: abstracting the submitted plan ID

`submit_plan()` returns a `PlanHandle` rather than a raw string. Before this change, each backend returned a different kind of string (a local root key, a Celery chord UUID, a Step Functions execution ARN) with no common interface.

`PlanHandle` is a Pydantic model — fully JSON-serialisable. Typical Django pattern:

```python
# In the view that kicks off a computation:
handle = backend.submit_plan(plan)
record.plan_handle = handle.to_json()
record.save()

# In a Celery beat task that polls for completion:
handle = PlanHandle.from_json(record.plan_handle)
state = handle.get_state()   # no S3 queries, no backend instance needed
if state in ("success", "failure"):
    record.state = state
    record.save()
```

### `get_state()` never queries S3

State is read from the native mechanism of each backend:
- **Local**: always `"success"` — the execution is synchronous and has already finished.
- **Celery**: `AsyncResult(plan_id, app=app).state` — hits the Celery result backend (Redis).
- **Step Functions**: `sfn.describe_execution(executionArn=plan_id)` — a single API call.

This keeps the Django API layer fast even when task results live in S3. The Celery result backend (Redis) and Step Functions both have sub-millisecond to low-millisecond response times.

---

## Progress: `PlanHandle.get_progress()` and `ProgressChecker`

### Why progress checking is in the storage layer

A node is complete when `manifest.json` exists at its storage prefix. This check is storage-specific (filesystem `os.path.exists` vs S3 `head_object`) and independent of the execution backend. It therefore belongs in the storage layer, not in the backends.

The `ProgressChecker` protocol (`muflow/storage/progress.py`) checks *multiple* prefixes at once. This is intentionally separate from `StorageBackend`, which is bound to a *single* prefix. The checkers are serialisable to a plain config dict so they can be reconstructed inside `PlanHandle` after deserialisation.

```
PlanHandle.storage_type + storage_config
    → make_progress_checker()
        → LocalProgressChecker  (os.path.exists per prefix)
        → S3ProgressChecker     (HEAD request per prefix)
```

Adding a new storage backend (e.g. GCS, Azure Blob) requires only:
1. A new `XxxProgressChecker` class with `completed_prefixes()`, `to_config()`, `from_config()`.
2. A new branch in `make_progress_checker()`.

`PlanHandle` and the backends do not change.

### Why S3 HEAD requests are acceptable

The API layer already knows the S3 bucket and key structure because it generates pre-signed URLs for delivering results directly to clients. This means S3 key structure is already part of the API contract — checking `manifest.json` adds no new coupling.

A `HEAD` request from within the same AWS region is 10–50 ms. For a plan with N nodes, `get_progress()` issues N sequential HEAD requests, which is acceptable for plans up to ~20–30 nodes polled at human-visible intervals (seconds). For larger plans or sub-second polling, a `RedisProgressChecker` can be added as a drop-in replacement without changing the interface.

### `node_breakdown` for fine-grained access

`PlanProgress.node_breakdown` is a `dict[str, bool]` mapping every node key to its completion status. This lets a caller check a specific node (e.g. "is the root node done?") without re-running the full check:

```python
progress = handle.get_progress()
if progress.node_breakdown[plan.root_key]:
    # root result is available — generate pre-signed URL
```

---

## Completion callbacks

### Homogenised signature

The original `CompletionCallback.notify(analysis_id: int, result: ExecutionResult)` embedded a domain concept (`analysis_id`) in the library. The signature is now:

```python
def notify(self, plan_id: str, success: bool, error: Optional[str]) -> None
```

The `plan_id` is the same value stored in `PlanHandle`. Callers that need to map it to a domain record (e.g. an `analysis_id`) maintain that mapping themselves — the library does not need to know about it.

### Why callbacks don't work for Step Functions

Step Functions executes fully asynchronously: `submit_plan()` returns immediately after calling `sfn.start_execution()`, and AWS drives the Lambda invocations from that point. There is no muflow process alive when nodes complete, so there is no place to call `callback.notify()`. The recommended approach is polling via `PlanHandle.get_state()` or `PlanHandle.get_progress()`, or setting up a CloudWatch EventBridge rule on Step Functions state-change events.

### Celery completion callback wiring

For Celery, completion notification is wired through a standard Celery mechanism: a `muflow.send_completion` task is registered by `create_celery_task()`. When a `CeleryCompletionCallback` is passed to `CeleryBackend.submit_plan()`, the outermost chord is wrapped with this task as its callback:

```
chord(all_plan_tasks, muflow.send_completion.si(plan_id, task_name, queue))
```

The `send_completion` task runs in a Celery worker on plan completion and calls `app.send_task(callback_task_name, args=[plan_id, True, None])`. Only `CeleryCompletionCallback` is accepted — passing any other implementation raises `TypeError` at `submit_plan()` time, because callbacks must be serialisable as Celery task arguments.

### The no-callback polling pattern

For the common Django use case (update a DB record when a plan finishes), no callback infrastructure is needed at all:

1. `submit_plan()` → store `handle.to_json()` in a model field.
2. A lightweight Celery beat task polls `PlanHandle.from_json(stored).get_state()` for all pending records, updates the DB when terminal.

This avoids the need to configure callback tasks, and works uniformly across all backends.

---

## Node-level callbacks and `LocalBackend`

`on_node_start`, `on_node_complete`, and `on_node_failure` are *not* part of the `ExecutionBackend` protocol. They are keyword-only parameters on `LocalBackend.submit_plan()` only.

**Why:** Async backends (Celery, Step Functions) have no mechanism to fire these synchronously back to the submitting process. Including them in the protocol creates false expectations — callers might pass them to a `CeleryBackend` expecting them to work, and nothing would happen. Keeping them local-only makes it explicit that they are a development and testing tool, not a production observation mechanism.

For production observation, use `PlanHandle.get_progress()` (storage polling) or `CeleryCompletionCallback` (plan-level Celery notification).

---

## Sentinel root node

When a `Pipeline` has multiple terminal steps (steps with no dependents), `build_plan()` inserts an invisible sentinel node with no-op function that depends on all of them. This ensures every plan has exactly one `root_key`, which simplifies:
- `TaskPlan.is_complete(completed)` — a single key check: `root_key in completed`.
- `PlanHandle.plan_id` — the local backend uses `root_key` as the plan ID.
- Progress checking — the root node's storage prefix can be checked to answer "is the final result available?".

---

## Future: Redis-based progress

`ProgressChecker` is designed to accommodate a `RedisProgressChecker` for deployments where S3 HEAD latency is a concern or real-time push (SSE/WebSocket) is needed.

For Celery deployments, Redis is already present as the broker. Workers would write `SADD muflow:completed:{plan_id} {prefix}` alongside `write_manifest()`. A `RedisProgressChecker` would replace N HEAD requests with a single `SMEMBERS` call.

For real-time push, the Django SSE endpoint can `SUBSCRIBE muflow:plan:{plan_id}` and stream events to the client as they arrive, rather than polling. This requires additional worker-side instrumentation but no changes to the `ProgressChecker` interface.
