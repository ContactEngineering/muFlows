"""Base execution backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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

    def submit(self, analysis_id: int, payload: dict) -> str:
        """Submit a workflow node for execution.

        Parameters
        ----------
        analysis_id : int
            Database ID of the WorkflowResult.
        payload : dict
            Backend-specific payload containing:
            - function: workflow function name
            - kwargs: workflow parameters
            - storage_prefix: where to write outputs
            - dependency_prefixes: dict of dependency key -> prefix
            - subject_data_key: S3 key for subject data (if applicable)
            - bucket: S3 bucket name
            - queue: task queue name (for Celery)
            - lambda_function: Lambda function name (for Lambda)

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
            Function that takes (analysis_id, payload) and executes the
            workflow. If not provided, submit() will raise NotImplementedError.
        """
        self._executor_fn = executor_fn
        self._states = {}  # task_id -> state

    def submit(self, analysis_id: int, payload: dict) -> str:
        """Execute workflow synchronously.

        Parameters
        ----------
        analysis_id : int
            Database ID of the WorkflowResult.
        payload : dict
            Execution payload.

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
