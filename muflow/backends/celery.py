"""Database-agnostic Celery execution backend.

This module provides a CeleryBackend that executes workflows without database
access, paralleling the LambdaBackend. Workers receive serialized ExecutionPayload
and use S3WorkflowContext for all I/O.

This enables:
- Same workflow code running on Celery, Lambda, or local without modification
- Horizontal scaling of Celery workers without Django/database dependencies
- Moving execution from Django-coupled Celery to database-agnostic Celery
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from muflow.executor import ExecutionPayload, ExecutionResult

_log = logging.getLogger(__name__)


class CeleryBackend:
    """Database-agnostic Celery backend for muFlow.

    Unlike Django's CeleryBackend which only passes analysis_id and looks up
    data from the database, this backend passes the full ExecutionPayload.
    Workers execute without database access using S3WorkflowContext.

    Parameters
    ----------
    celery_app
        Celery application instance.
    bucket : str
        S3 bucket for workflow I/O.
    default_queue : str
        Default Celery queue name. Defaults to "default".
    task_name : str
        Name of the Celery task to invoke. Defaults to "muflow.execute_workflow_task".

    Example
    -------
    >>> from celery import Celery
    >>> from muflow import ExecutionPayload
    >>>
    >>> app = Celery("myapp")
    >>> backend = CeleryBackend(
    ...     celery_app=app,
    ...     bucket="my-bucket",
    ...     default_queue="workflows",
    ... )
    >>>
    >>> payload = ExecutionPayload(
    ...     workflow_name="sds_ml.v3.gpr.training",
    ...     kwargs={"threshold": 0.5},
    ...     storage_prefix="muflow/gpr/abc123",
    ...     allowed_outputs={"result.json", "model.nc"},
    ... )
    >>> task_id = backend.submit(analysis_id=123, payload=payload)
    """

    def __init__(
        self,
        celery_app,
        bucket: str,
        default_queue: str = "default",
        task_name: str = "muflow.execute_workflow_task",
    ):
        self._app = celery_app
        self._bucket = bucket
        self._default_queue = default_queue
        self._task_name = task_name

    def submit(self, analysis_id: int, payload: ExecutionPayload) -> str:
        """Submit workflow for Celery execution.

        The full ExecutionPayload is serialized and passed to the worker.
        No database lookup is performed by the worker.

        Parameters
        ----------
        analysis_id : int
            ID to associate with this execution. Used for callbacks
            and tracking, but not for database lookups.
        payload : ExecutionPayload
            Complete workflow execution payload.

        Returns
        -------
        str
            Celery task ID.
        """
        queue = payload.queue or self._default_queue

        task = self._app.send_task(
            self._task_name,
            args=[analysis_id, payload.to_dict(), self._bucket],
            queue=queue,
        )

        _log.debug(
            f"Submitted task {task.id} for analysis {analysis_id} "
            f"to queue {queue} (workflow: {payload.workflow_name})"
        )
        return task.id

    def cancel(self, task_id: str) -> None:
        """Cancel a running task.

        Parameters
        ----------
        task_id : str
            Celery task ID to cancel.
        """
        self._app.control.revoke(task_id, terminate=True)
        _log.debug(f"Cancelled task {task_id}")

    def get_state(self, task_id: str) -> str:
        """Get the state of a task.

        Parameters
        ----------
        task_id : str
            Celery task ID.

        Returns
        -------
        str
            Task state. Maps Celery states to muflow states:
            - PENDING -> "pending"
            - STARTED -> "running"
            - SUCCESS -> "success"
            - FAILURE -> "failure"
            - REVOKED -> "cancelled"
        """
        from celery.result import AsyncResult

        result = AsyncResult(task_id, app=self._app)
        state = result.state

        # Map Celery states to muflow states
        state_map = {
            "PENDING": "pending",
            "STARTED": "running",
            "SUCCESS": "success",
            "FAILURE": "failure",
            "REVOKED": "cancelled",
            "RETRY": "pending",
        }
        return state_map.get(state, "pending")


def create_celery_task(
    celery_app,
    workflow_registry: dict,
    on_complete: Optional[Callable[[int, ExecutionResult], None]] = None,
    task_name: str = "muflow.execute_workflow_task",
):
    """Create a Celery task for database-agnostic workflow execution.

    This is a factory function that creates a Celery task configured
    with a registry of available workflow implementations. The task
    uses S3WorkflowContext for all I/O, with no database access.

    Similar to create_lambda_handler() but for Celery workers.

    Parameters
    ----------
    celery_app
        Celery application instance.
    workflow_registry : dict
        Mapping from workflow name to implementation class.
    on_complete : callable, optional
        Function called with (analysis_id, ExecutionResult) after execution.
        This is typically used to trigger a completion callback.
    task_name : str
        Name for the Celery task. Defaults to "muflow.execute_workflow_task".

    Returns
    -------
    celery.Task
        The registered Celery task.

    Example
    -------
    >>> from celery import Celery
    >>> from myworkflows import GPRWorkflow, GPCWorkflow
    >>>
    >>> app = Celery("myapp")
    >>> task = create_celery_task(
    ...     celery_app=app,
    ...     workflow_registry={
    ...         "sds_ml.v3.gpr.training": GPRWorkflow,
    ...         "sds_ml.v3.gpc.training": GPCWorkflow,
    ...     },
    ... )

    Worker Configuration
    --------------------
    In the worker's startup module, register the task before starting:

    >>> from celery import Celery
    >>> from muflow.backends.celery_backend import create_celery_task
    >>> from myworkflows import workflow_registry
    >>>
    >>> app = Celery("worker")
    >>> app.config_from_object("myconfig")
    >>>
    >>> # Register the task
    >>> create_celery_task(app, workflow_registry)
    >>>
    >>> # Now workers can receive muflow.execute_workflow_task
    """
    from muflow.context import S3WorkflowContext
    from muflow.executor import execute_workflow

    @celery_app.task(name=task_name, bind=True)
    def execute_workflow_task(
        self,
        analysis_id: int,
        payload_dict: dict,
        bucket: str,
    ):
        """Execute a workflow without database access.

        Parameters
        ----------
        self : celery.Task
            Bound Celery task instance.
        analysis_id : int
            ID for tracking and callbacks (not used for DB lookup).
        payload_dict : dict
            Serialized ExecutionPayload.
        bucket : str
            S3 bucket for I/O.

        Returns
        -------
        dict
            Execution result summary.
        """
        # Reconstruct payload from serialized dict
        payload = ExecutionPayload.from_dict(payload_dict)

        _log.info(
            f"execute_workflow_task: Starting {payload.workflow_name} "
            f"(analysis_id={analysis_id}, task_id={self.request.id})"
        )

        if payload.workflow_name not in workflow_registry:
            error_msg = f"Unknown workflow: {payload.workflow_name}"
            _log.error(f"execute_workflow_task: {error_msg}")
            result = ExecutionResult(
                success=False,
                error_message=error_msg,
            )
            if on_complete:
                on_complete(analysis_id, result)
            raise ValueError(error_msg)

        # Create S3 context (same as Lambda handler)
        ctx = S3WorkflowContext(
            storage_prefix=payload.storage_prefix,
            kwargs=payload.kwargs,
            dependency_prefixes=payload.dependency_prefixes,
            bucket=bucket,
            allowed_outputs=payload.allowed_outputs,
        )

        # Execute using the pure execution function
        result = execute_workflow(
            payload=payload,
            context=ctx,
            get_implementation=lambda name: workflow_registry[name],
        )

        # Write execution result to S3 for verification/debugging
        try:
            ctx.save_json("_execution_result.json", {
                "analysis_id": analysis_id,
                "task_id": self.request.id,
                "success": result.success,
                "error_message": result.error_message,
                "files_written": result.files_written,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            _log.warning(f"Failed to write _execution_result.json: {e}")

        # Trigger completion callback
        if on_complete:
            try:
                on_complete(analysis_id, result)
            except Exception as e:
                _log.exception(f"Completion callback failed: {e}")

        if result.success:
            _log.info(
                f"execute_workflow_task: Completed {payload.workflow_name} "
                f"(analysis_id={analysis_id})"
            )
        else:
            _log.error(
                f"execute_workflow_task: Failed {payload.workflow_name} "
                f"(analysis_id={analysis_id}): {result.error_message}"
            )
            # Re-raise to mark task as failed in Celery
            raise RuntimeError(result.error_message)

        return {
            "analysis_id": analysis_id,
            "success": result.success,
            "files_written": result.files_written,
        }

    return execute_workflow_task
