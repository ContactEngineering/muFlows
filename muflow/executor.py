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
from dataclasses import dataclass, field
from typing import Callable, Optional

from muflow.context import WorkflowContext


@dataclass
class ExecutionPayload:
    """Serializable input for workflow execution.

    Contains all information needed to execute a workflow without
    any database access.

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
    allowed_outputs : set[str]
        Set of filenames this workflow is allowed to write.
    """

    workflow_name: str
    kwargs: dict
    storage_prefix: str
    dependency_prefixes: dict[str, str] = field(default_factory=dict)
    allowed_outputs: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "workflow_name": self.workflow_name,
            "kwargs": self.kwargs,
            "storage_prefix": self.storage_prefix,
            "dependency_prefixes": self.dependency_prefixes,
            "allowed_outputs": list(self.allowed_outputs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPayload:
        """Create from dictionary."""
        return cls(
            workflow_name=data["workflow_name"],
            kwargs=data["kwargs"],
            storage_prefix=data["storage_prefix"],
            dependency_prefixes=data.get("dependency_prefixes", {}),
            allowed_outputs=set(data.get("allowed_outputs", [])),
        )


@dataclass
class ExecutionResult:
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
        List of output files that were written.
    """

    success: bool
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    files_written: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "files_written": self.files_written,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionResult:
        """Create from dictionary."""
        return cls(
            success=data["success"],
            error_message=data.get("error_message"),
            error_traceback=data.get("error_traceback"),
            files_written=data.get("files_written", []),
        )


def execute_workflow(
    payload: ExecutionPayload,
    context: WorkflowContext,
    get_implementation: Callable[[str], type],
) -> ExecutionResult:
    """Execute a workflow. Pure function with no database access.

    This is the core execution function used by all backends. It:
    1. Looks up the workflow implementation by name
    2. Instantiates it with the provided kwargs
    3. Calls the execute() method with the context
    4. Returns success/failure result

    Parameters
    ----------
    payload : ExecutionPayload
        Workflow name, kwargs, and storage configuration.
    context : WorkflowContext
        File I/O interface. Must be configured with:
        - storage_prefix matching payload.storage_prefix
        - dependency_prefixes matching payload.dependency_prefixes
        - allowed_outputs matching payload.allowed_outputs
    get_implementation : Callable[[str], type]
        Function that returns the implementation class for a workflow name.
        This allows domain-specific registries to be passed in.

    Returns
    -------
    ExecutionResult
        Success status, any error information, and list of files written.

    Example
    -------
    >>> # In a Celery task:
    >>> context = DjangoWorkflowContext(analysis, dependencies, allowed_outputs)
    >>> result = execute_workflow(payload, context, get_workflow_registry)
    >>>
    >>> # In a Lambda handler:
    >>> context = S3WorkflowContext(
    ...     storage_prefix=payload.storage_prefix,
    ...     kwargs=payload.kwargs,
    ...     dependency_prefixes=payload.dependency_prefixes,
    ...     bucket=bucket,
    ...     allowed_outputs=payload.allowed_outputs,
    ... )
    >>> result = execute_workflow(payload, context, get_workflow_registry)
    """
    try:
        # Get the implementation class
        impl_class = get_implementation(payload.workflow_name)

        # Instantiate with kwargs
        impl = impl_class(**payload.kwargs)

        # Execute the workflow
        impl.execute(context)

        return ExecutionResult(
            success=True,
            files_written=list(payload.allowed_outputs),
        )
    except Exception as exc:
        return ExecutionResult(
            success=False,
            error_message=str(exc),
            error_traceback=traceback.format_exc(),
        )
