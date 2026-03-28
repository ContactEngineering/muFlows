"""Pure workflow execution.

This module provides the core execution function that is completely
database-agnostic. The same function can be called from:
- Celery tasks (with DjangoWorkflowContext)
- AWS Lambda handlers (with S3WorkflowContext)
- AWS Batch jobs (with S3WorkflowContext)
- Local testing (with LocalFolderContext)

The execution layer has no knowledge of Django, database models, or
TopoBank-specific concepts like "subjects". All domain-specific logic
is handled by the calling layer before invoking execute_workflow().
"""

from __future__ import annotations

import traceback
from typing import Callable, Optional

import pydantic

from muflow.context import WorkflowContext


class ExecutionPayload(pydantic.BaseModel):
    """Serializable input for workflow execution.

    Contains all information needed to execute a workflow without
    any database access, plus optional routing information.

    Attributes
    ----------
    workflow_name : str
        Name of the workflow implementation to run.
    kwargs : dict
        Parameters to pass to the workflow.
    storage_prefix : str
        S3 key prefix or local path for output files.
    dependency_prefixes : dict[str, str]
        Mapping from dependency key to storage prefix.
    queue : str | None
        Optional queue name for routing. Used by backends that support
        multiple queues (e.g., Celery). If None, backend uses its default.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    workflow_name: str
    kwargs: dict
    storage_prefix: str
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
    """Serializable output from workflow execution.

    Attributes
    ----------
    success : bool
        Whether execution completed without error.
    error_message : str | None
        Error message if execution failed.
    error_traceback : str | None
        Full traceback if execution failed.
    files_written : list[str]
        List of output files that were written (from storage backend manifest).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    success: bool
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    files_written: list[str] = []

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionResult:
        """Create from dictionary."""
        return cls.model_validate(data)


def execute_workflow(
    payload: ExecutionPayload,
    context: WorkflowContext,
    get_entry: Callable,
) -> ExecutionResult:
    """Execute a workflow.  Pure function with no database access.

    This is the core execution function used by all backends.  It:
    1. Looks up the ``WorkflowEntry`` by name
    2. Validates kwargs against the entry's ``parameters`` model (if any)
       and stores the result on ``context._parameters``
    3. Calls ``entry.fn(context)``
    4. Writes ``manifest.json`` via the storage backend (always, even on error)
    5. Returns success/failure result with the list of files written

    Parameters
    ----------
    payload : ExecutionPayload
        Workflow name, kwargs, and storage configuration.
    context : WorkflowContext
        Execution context wrapping a storage backend.
    get_entry : Callable[[str], WorkflowEntry]
        Function that returns a ``WorkflowEntry`` for a workflow name.
        Typically ``lambda name: registry.get_all()[name]``.

    Returns
    -------
    ExecutionResult
        Success status, any error information, and list of files written.
    """
    from muflow.registry import WorkflowEntry

    try:
        # Look up the workflow entry
        entry = get_entry(payload.workflow_name)

        # If get_entry returned a class (legacy caller), wrap it
        if not isinstance(entry, WorkflowEntry):
            klass = entry
            def _legacy_fn(ctx):
                impl = klass(**payload.kwargs)
                return impl.execute(ctx)
            entry = WorkflowEntry(
                name=payload.workflow_name, fn=_legacy_fn
            )

        # Validate parameters and attach to context
        if entry.parameters is not None:
            context._parameters = entry.parameters(**payload.kwargs)
        else:
            context._parameters = None

        # Execute the workflow
        entry.fn(context)

        return ExecutionResult(
            success=True,
            files_written=sorted(context.storage.written_files),
        )
    except Exception as exc:
        return ExecutionResult(
            success=False,
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
            files_written=sorted(context.storage.written_files),
        )
    finally:
        # Always write the manifest, even on error
        context.storage.write_manifest()
