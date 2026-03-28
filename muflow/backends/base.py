"""Base execution backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from muflow.executor import ExecutionPayload


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for workflow execution backends.

    An execution backend knows how to:
    - Submit a workflow node for execution
    - Cancel a running task
    - Query task state

    Backends do NOT handle:
    - Dependency resolution (done by WorkflowPlanner)
    - State management (done by Django/PlanExecutor)
    - File I/O (done by WorkflowContext)

    Implementations:
    - CeleryBackend: Dispatches to Celery workers (in topobank, not here)
    - LambdaBackend: Invokes AWS Lambda functions
    - BatchBackend: Submits AWS Batch jobs (future)
    - LocalBackend: Runs synchronously (for testing)
    """

    def submit(self, analysis_id: int, payload: "ExecutionPayload") -> str:
        """Submit a workflow node for execution.

        Parameters
        ----------
        analysis_id : int
            Database ID of the WorkflowResult. Used for tracking and
            callbacks after execution completes.
        payload : ExecutionPayload
            Workflow execution payload containing:
            - workflow_name: name of workflow to execute
            - kwargs: workflow parameters
            - storage_prefix: where to write outputs
            - dependency_prefixes: dict of dependency key -> prefix

        Returns
        -------
        str
            Backend-specific task ID for tracking.
        """
        ...

    def cancel(self, task_id: str) -> None:
        """Cancel a running task.

        Parameters
        ----------
        task_id : str
            Task ID returned by submit().
        """
        ...

    def get_state(self, task_id: str) -> str:
        """Query task state.

        Parameters
        ----------
        task_id : str
            Task ID returned by submit().

        Returns
        -------
        str
            One of: "pending", "running", "success", "failure", "cancelled"
        """
        ...


class LocalBackend:
    """Synchronous local execution backend for testing.

    Executes workflows in the current process, blocking until complete.
    """

    def __init__(self, executor_fn=None):
        """Initialize the local backend.

        Parameters
        ----------
        executor_fn : callable, optional
            Function that takes (analysis_id, payload: ExecutionPayload) and
            executes the workflow. If not provided, submit() will raise
            NotImplementedError.
        """
        self._executor_fn = executor_fn
        self._states = {}  # task_id -> state

    def submit(self, analysis_id: int, payload: "ExecutionPayload") -> str:
        """Execute workflow synchronously.

        Parameters
        ----------
        analysis_id : int
            Database ID of the WorkflowResult.
        payload : ExecutionPayload
            Workflow execution payload.

        Returns
        -------
        str
            Task ID (just the analysis_id as string for local execution).
        """
        task_id = str(analysis_id)

        if self._executor_fn is None:
            raise NotImplementedError(
                "LocalBackend requires an executor_fn to be provided"
            )

        self._states[task_id] = "running"
        try:
            self._executor_fn(analysis_id, payload)
            self._states[task_id] = "success"
        except Exception:
            self._states[task_id] = "failure"
            raise

        return task_id

    def cancel(self, task_id: str) -> None:
        """Cancel not supported for synchronous execution."""
        raise NotImplementedError("LocalBackend does not support cancellation")

    def get_state(self, task_id: str) -> str:
        """Get task state."""
        return self._states.get(task_id, "pending")
