"""Completion callback protocols and implementations.

Callbacks notify the calling application when a plan finishes, so that
it can update state (e.g. a database record) without the workflow itself
requiring database access.

All implementations share the same signature::

    notify(plan_id: str, success: bool, error: Optional[str]) -> None

The ``plan_id`` matches the value stored in :class:`PlanHandle` and
returned by ``submit_plan()``. Callers that need to map it back to a
domain object (e.g. an ``analysis_id``) maintain that mapping themselves.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

_log = logging.getLogger(__name__)


class CompletionCallback(Protocol):
    """Protocol for plan-level completion notifications."""

    def notify(
        self,
        plan_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Notify that a plan completed.

        Parameters
        ----------
        plan_id : str
            ID of the plan that completed (matches :attr:`PlanHandle.plan_id`).
        success : bool
            Whether the plan completed successfully.
        error : str, optional
            Error message if the plan failed.
        """
        ...


class CeleryCompletionCallback:
    """Trigger a Celery task when a plan completes.

    This is the recommended callback for async (Celery) backends. On
    completion, it dispatches a named Celery task with
    ``(plan_id, success, error)`` as arguments.  The receiving task
    typically has Django/database access to update records.

    Parameters
    ----------
    celery_app
        Celery application instance.
    task_name : str
        Name of the Celery task to call on completion.
    queue : str
        Queue to send the callback task to. Defaults to ``"callbacks"``.

    Example
    -------
    >>> callback = CeleryCompletionCallback(
    ...     celery_app=app,
    ...     task_name="myapp.tasks.on_plan_complete",
    ...     queue="callbacks",
    ... )
    >>> backend.submit_plan(plan, completion_callback=callback)
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

    def notify(
        self,
        plan_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Dispatch the completion Celery task.

        Parameters
        ----------
        plan_id : str
            ID of the completed plan.
        success : bool
            Whether the plan succeeded.
        error : str, optional
            Error message on failure.
        """
        self._app.send_task(
            self._task_name,
            args=[plan_id, success, error],
            queue=self._queue,
        )
        _log.debug(
            f"Sent completion callback for plan {plan_id} "
            f"to {self._task_name} on queue {self._queue}"
        )


class NoOpCompletionCallback:
    """No-op callback for testing or when completion handling is not needed."""

    def notify(
        self,
        plan_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        pass


class LoggingCompletionCallback:
    """Log plan completion for debugging.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. Defaults to this module's logger.
    """

    def __init__(self, logger: logging.Logger = None):
        self._log = logger or _log

    def notify(
        self,
        plan_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        if success:
            self._log.info(f"Plan completed: plan_id={plan_id}")
        else:
            self._log.error(f"Plan failed: plan_id={plan_id}, error={error}")
