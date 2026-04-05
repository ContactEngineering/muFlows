"""Pure task execution.

This module provides the core execution function that is completely
database-agnostic. The same function can be called from:
- Celery tasks (with TaskContext + S3StorageBackend)
- AWS Lambda handlers (with TaskContext + S3StorageBackend)
- AWS Batch jobs (with TaskContext + S3StorageBackend)
- Local testing (with create_local_context)

The execution layer has no knowledge of Django, database models, or
TopoBank-specific concepts like "subjects". All domain-specific logic
is handled by the calling layer before invoking execute_task().
"""

from __future__ import annotations

import traceback
from typing import Callable, Optional

import pydantic

from muflow.context import TaskContext


class ExecutionPayload(pydantic.BaseModel):
    """Serializable input for task execution.

    Contains all information needed to execute a task without
    any database access, plus optional routing information.

    Attributes
    ----------
    task_name : str
        Name of the task implementation to run.
    kwargs : dict
        Parameters to pass to the task.
    storage_prefix : str
        S3 key prefix or local path for output files.
    context_data : dict
        Domain-specific identity/loading data (stored as context.json).
    dependency_prefixes : dict[str, str]
        Mapping from dependency key to storage prefix.
    queue : str | None
        Optional queue name for routing. Used by backends that support
        multiple queues (e.g., Celery). If None, backend uses its default.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    task_name: str
    kwargs: dict
    storage_prefix: str
    context_data: dict = {}
    dependency_prefixes: dict[str, str] = {}
    queue: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPayload:
        """Create from dictionary."""
        return cls.model_validate(data)


class ExecutionResult(pydantic.BaseModel):
    """Serializable output from task execution.

    Attributes
    ----------
    success : bool
        Whether execution completed without error.
    cached : bool
        True if results already existed and execution was skipped.
    error_message : str | None
        Error message if execution failed.
    error_traceback : str | None
        Full traceback if execution failed.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    success: bool
    cached: bool = False
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionResult:
        """Create from dictionary."""
        return cls.model_validate(data)


def execute_task(
    payload: ExecutionPayload,
    context: TaskContext,
    get_entry: Callable,
) -> ExecutionResult:
    """Execute a task.  Pure function with no database access.

    This is the core execution function used by all backends.  It:
    1. Looks up the ``TaskEntry`` by name
    2. Writes ``context.json`` to storage (from payload.context_data)
    3. Validates kwargs against the entry's ``parameters`` model (if any)
       and stores the result on ``context._parameters``
    4. Calls ``entry.fn(context)``
    5. Writes ``manifest.json`` via the storage backend (always, even on error)
    6. Returns success/failure result with the list of files written

    Parameters
    ----------
    payload : ExecutionPayload
        Task name, kwargs, and storage configuration.
    context : TaskContext
        Execution context wrapping a storage backend.
    get_entry : Callable[[str], TaskEntry]
        Function that returns a ``TaskEntry`` for a task name.
        Typically ``lambda name: registry.get_all()[name]``.

    Returns
    -------
    ExecutionResult
        Success/failure status and any error information.
    """
    from muflow.registry import TaskEntry

    # Early exit: if results already exist at this prefix, skip execution
    if context.storage.is_cached():
        return ExecutionResult(success=True, cached=True)

    try:
        # Write context.json (protected, so use internal storage method if possible)
        # For now, we assume the backend allows the executor to write it
        # via a side-channel or by relaxing PROTECTED_FILES for this call.
        context.storage.save_json(
            "context.json", payload.context_data, allow_protected=True
        )

        # Look up the task entry
        entry = get_entry(payload.task_name)

        if not isinstance(entry, TaskEntry):
            raise TypeError(f"Registry returned non-TaskEntry: {type(entry)}")

        # Validate parameters and attach to context
        if entry.parameters is not None:
            context._kwargs = entry.parameters(**payload.kwargs)
        else:
            context._kwargs = payload.kwargs

        # Execute the task
        entry.fn(context)

        return ExecutionResult(success=True)
    except Exception as exc:
        return ExecutionResult(
            success=False,
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
        )
    finally:
        # Always write the manifest, even on error
        context.storage.write_manifest()
