"""Completion callback protocols and implementations.

This module provides callback mechanisms for notifying when a task
completes. Callbacks are triggered after task execution, regardless
of success or failure.

The callback pattern allows the orchestration layer (e.g., Django) to
receive notifications from database-agnostic workers without the workers
needing database access.
"""

from __future__ import annotations

import logging
from typing import Protocol

from muflow.executor import ExecutionResult

_log = logging.getLogger(__name__)


class CompletionCallback(Protocol):
    """Protocol for completion notifications.

    Implementations receive notification when a task completes,
    allowing them to update state, trigger downstream tasks, etc.
    """

    def notify(self, analysis_id: int, result: ExecutionResult) -> None:
        """Notify that a task completed.

        Parameters
        ----------
        analysis_id : int
            ID of the analysis that completed.
        result : ExecutionResult
            Execution result with success status and any error information.
        """
        ...


class CeleryCompletionCallback:
    """Trigger a Celery task on completion.

    This is the primary callback mechanism for muFlow on Celery. When a
    task completes, it sends a Celery task to a specified queue.
    The callback task typically has Django/database access to update
    the TaskResult state.

    Parameters
    ----------
    celery_app
        Celery application instance.
    task_name : str
        Name of the Celery task to trigger on completion.
    queue : str
        Queue to send the callback task to. Defaults to "callbacks".

    Example
    -------
    >>> from celery import Celery
    >>>
    >>> app = Celery("myapp")
    >>> callback = CeleryCompletionCallback(
    ...     celery_app=app,
    ...     task_name="topobank.analysis.tasks.on_task_complete",
    ...     queue="callbacks",
    ... )
    >>>
    >>> # When used with CeleryBackend:
    >>> from muflow.backends.celery_backend import create_celery_task
    >>>
    >>> task = create_celery_task(
    ...     celery_app=app,
    ...     task_registry={...},
    ...     on_complete=callback.notify,
    ... )
    """

    def __init__(
        self,
        celery_app,
        task_name: str,
        queue: str = "callbacks",
    ):
        self._app = celery_app
        self._task_name = task_name
        self._queue = queue

    def notify(self, analysis_id: int, result: ExecutionResult) -> None:
        """Send completion notification as a Celery task.

        Parameters
        ----------
        analysis_id : int
            ID of the analysis that completed.
        result : ExecutionResult
            Execution result.
        """
        self._app.send_task(
            self._task_name,
            args=[analysis_id, result.to_dict()],
            queue=self._queue,
        )
        _log.debug(
            f"Sent completion callback for analysis {analysis_id} "
            f"to {self._task_name} on queue {self._queue}"
        )


class NoOpCompletionCallback:
    """No-op callback for testing or when caller handles completion.

    Use this when:
    - Testing without callback infrastructure
    - The caller handles completion explicitly (e.g., synchronous execution)
    - Callbacks are not needed for the use case
    """

    def notify(self, analysis_id: int, result: ExecutionResult) -> None:
        """Do nothing.

        Parameters
        ----------
        analysis_id : int
            Ignored.
        result : ExecutionResult
            Ignored.
        """
        pass


class LoggingCompletionCallback:
    """Log completion for debugging/testing.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. Defaults to this module's logger.
    """

    def __init__(self, logger: logging.Logger = None):
        self._log = logger or _log

    def notify(self, analysis_id: int, result: ExecutionResult) -> None:
        """Log the completion.

        Parameters
        ----------
        analysis_id : int
            ID of the analysis that completed.
        result : ExecutionResult
            Execution result.
        """
        if result.success:
            self._log.info(
                f"Task completed: analysis_id={analysis_id}, "
                f"files_written={result.files_written}"
            )
        else:
            self._log.error(
                f"Task failed: analysis_id={analysis_id}, "
                f"error={result.error_message}"
            )
