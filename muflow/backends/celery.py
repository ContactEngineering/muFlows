"""Celery execution backend with parallel DAG orchestration.

This module provides a CeleryBackend that executes task plans using
Celery's chord and group primitives for parallel execution.

Architecture
------------
- submit_plan() converts the TaskPlan DAG into Celery chord/group structures
- Nodes at the same "level" (same dependency depth) run in parallel via group()
- Levels are chained together via chord() so each level waits for the previous
- Workers execute individual nodes via execute_task()

Example DAG
-----------
    feature_0 ─┐
    feature_1 ─┼─► training
    feature_2 ─┘
               └─► loo_fold_0
               └─► loo_fold_1

Becomes:
    chord(
        group(feature_0, feature_1, feature_2),  # Level 0: parallel
        group(training, loo_0, loo_1),           # Level 1: parallel, after level 0
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from muflow.backends.callbacks import CompletionCallback
    from muflow.backends.handle import PlanHandle
    from muflow.plan import TaskNode, TaskPlan

_log = logging.getLogger(__name__)


class CeleryBackend:
    """Celery backend with parallel DAG orchestration.

    Converts TaskPlan into Celery chord/group structures for efficient
    parallel execution. Nodes at the same dependency level run in parallel.

    Parameters
    ----------
    celery_app
        Celery application instance.
    bucket : str
        S3 bucket for task I/O.
    base_prefix : str
        S3 key prefix used when building the plan.
        Default: ``"muflow"``.
    task_name : str
        Name of the Celery task for node execution.
        Defaults to "muflow.execute_node".

    Example
    -------
    >>> from celery import Celery
    >>> from muflow.backends import CeleryBackend
    >>>
    >>> app = Celery("myapp")
    >>> backend = CeleryBackend(app, bucket="my-bucket", base_prefix="muflow")
    >>>
    >>> plan = my_pipeline.build_plan("tag:1", kwargs, base_prefix="muflow")
    >>> plan_id = backend.submit_plan(plan)
    """

    def __init__(
        self,
        celery_app,
        bucket: str,
        base_prefix: str = "muflow",
        task_name: str = "muflow.execute_node",
    ):
        self._app = celery_app
        self._bucket = bucket
        self._base_prefix = base_prefix
        self._task_name = task_name
        self._plan_results: dict[str, object] = {}  # plan_id -> AsyncResult

    def submit_plan(
        self,
        plan: "TaskPlan",
        completion_callback: Optional["CompletionCallback"] = None,
    ) -> "PlanHandle":
        """Submit a task plan for parallel execution.

        Converts the DAG into Celery chord/group structures:
        - Nodes with no dependencies (level 0) run first in parallel
        - Nodes at level N run after all level N-1 nodes complete
        - Within each level, nodes run in parallel

        Parameters
        ----------
        plan : TaskPlan
            Complete task plan.
        completion_callback : CompletionCallback, optional
            Called when the plan completes. Must be a
            :class:`CeleryCompletionCallback` for async dispatch; passing
            any other type raises ``TypeError``.

        Returns
        -------
        PlanHandle
            Handle with backend="celery".
        """
        from muflow.backends.callbacks import CeleryCompletionCallback
        from muflow.backends.handle import PlanHandle

        if completion_callback is not None and not isinstance(
            completion_callback, CeleryCompletionCallback
        ):
            raise TypeError(
                "CeleryBackend requires a CeleryCompletionCallback for async "
                "dispatch. Pass a CeleryCompletionCallback or use PlanHandle "
                "polling instead."
            )

        levels = self._compute_levels(plan)

        _log.info(
            f"Submitting plan {plan.root_key[:16]}... with {len(plan.nodes)} nodes "
            f"in {len(levels)} levels"
        )

        celery_task = self._build_celery_task(levels, plan)

        # Wrap with completion notification if requested
        if completion_callback is not None:
            celery_task = self._wrap_with_completion(
                celery_task, plan.root_key, completion_callback
            )

        result = celery_task.apply_async()
        self._plan_results[result.id] = result

        _log.info(f"Submitted plan as Celery task {result.id}")
        return PlanHandle(
            backend="celery",
            plan_id=result.id,
            node_prefixes={k: n.storage_prefix for k, n in plan.nodes.items()},
            storage_type="s3",
            storage_config={"bucket": self._bucket},
        )

    def get_plan_state(self, plan_id: str) -> str:
        """Get the state of a plan execution.

        Parameters
        ----------
        plan_id : str
            Plan execution ID (Celery task ID).

        Returns
        -------
        str
            One of: "pending", "running", "success", "failure"
        """
        result = self._plan_results.get(plan_id)
        if result is None:
            # Try to get from Celery
            from celery.result import AsyncResult
            result = AsyncResult(plan_id, app=self._app)

        state_map = {
            "PENDING": "pending",
            "STARTED": "running",
            "SUCCESS": "success",
            "FAILURE": "failure",
            "REVOKED": "failure",
        }

        # GroupResult (from group/chord) doesn't have .state;
        # check if all children completed instead.
        if not hasattr(result, "state"):
            if result.ready():
                return "failure" if result.failed() else "success"
            return "running"

        return state_map.get(result.state, "pending")

    def cancel_plan(self, plan_id: str) -> None:
        """Cancel a running plan.

        Note: This revokes the top-level task. Individual node tasks
        that are already running may continue.

        Parameters
        ----------
        plan_id : str
            Plan execution ID (Celery task ID).
        """
        self._app.control.revoke(plan_id, terminate=True)
        _log.info(f"Cancelled plan {plan_id}")

    def _compute_levels(self, plan: "TaskPlan") -> list[list["TaskNode"]]:
        """Group nodes by execution level (topological sort).

        Level 0: nodes with no dependencies (leaf nodes)
        Level N: nodes whose dependencies are all in levels < N

        Parameters
        ----------
        plan : TaskPlan
            The task plan.

        Returns
        -------
        list[list[TaskNode]]
            Nodes grouped by execution level.
        """
        levels: list[list["TaskNode"]] = []
        remaining = set(plan.nodes.keys())
        completed: set[str] = set()

        while remaining:
            ready = [
                plan.nodes[key]
                for key in remaining
                if all(d in completed for d in plan.nodes[key].depends_on)
            ]

            if not ready:
                raise ValueError(
                    f"Circular dependency detected. Remaining nodes: {remaining}"
                )

            levels.append(ready)
            completed.update(n.key for n in ready)
            remaining -= {n.key for n in ready}

        return levels

    def _build_celery_task(
        self,
        levels: list[list["TaskNode"]],
        plan: "TaskPlan",
    ):
        """Convert execution levels into Celery chord/chain structure.

        Parameters
        ----------
        levels : list[list[TaskNode]]
            Nodes grouped by execution level.
        plan : TaskPlan
            The task plan (for dependency prefixes).

        Returns
        -------
        celery.canvas.Signature
            Celery task signature.
        """
        from celery import chord, group

        celery_groups = []
        for level in levels:
            tasks = [self._make_node_task(node, plan) for node in level]
            celery_groups.append(group(tasks))

        if len(celery_groups) == 1:
            return celery_groups[0]

        # Build from the end backwards:
        # chord(g0, chord(g1, chord(g2, g3)))
        result = celery_groups[-1]
        for grp in reversed(celery_groups[:-1]):
            result = chord(grp, result)

        return result

    def _wrap_with_completion(
        self,
        celery_task,
        plan_id: str,
        completion_callback: "CompletionCallback",
    ):
        """Wrap the task canvas with a completion notification chord.

        The notification task (``muflow.send_completion``) is called after
        the entire canvas finishes.  It dispatches the
        :class:`CeleryCompletionCallback`'s target task.

        Parameters
        ----------
        celery_task
            The already-built Celery canvas.
        plan_id : str
            Plan root key, passed to the callback as ``plan_id``.
        completion_callback : CeleryCompletionCallback
            Callback to fire on success.

        Returns
        -------
        celery.canvas.Signature
            Wrapped canvas.
        """
        from celery import chord

        notify_sig = self._app.signature(
            "muflow.send_completion",
            args=[plan_id, completion_callback._task_name, completion_callback._queue],
            immutable=True,
            queue=completion_callback._queue,
        )
        return chord(celery_task, notify_sig)

    def _make_node_task(self, node: "TaskNode", plan: "TaskPlan"):
        """Create a Celery task signature for a node.

        Parameters
        ----------
        node : TaskNode
            The node to create a task for.
        plan : TaskPlan
            The task plan (for dependency prefixes).

        Returns
        -------
        celery.canvas.Signature
            Celery task signature.
        """
        # Use pre-computed dependency access map from plan
        dependency_prefixes = node.dependency_access_map

        # Build payload dict
        payload_dict = {
            "task_name": node.function,
            "kwargs": node.kwargs,
            "storage_prefix": node.storage_prefix,
            "dependency_prefixes": dependency_prefixes,
        }

        # Get queue from node or use default
        queue = getattr(node, 'queue', None) or "default"

        # Create immutable task signature (prevents chord from injecting
        # the header-group result as an extra positional argument)
        return self._app.signature(
            self._task_name,
            args=[node.key, payload_dict, self._bucket],
            queue=queue,
            immutable=True,
        )


def create_celery_task(
    celery_app,
    task_registry: Optional[dict] = None,
    task_name: str = "muflow.execute_node",
) -> object:
    """Create a Celery task for executing task nodes.

    This is a factory function that creates a Celery task configured
    with a registry of available task implementations. The task
    is called by CeleryBackend for each node in the plan.

    Parameters
    ----------
    celery_app
        Celery application instance.
    task_registry : dict, optional
        Mapping from task name to TaskEntry (or legacy class).
        Defaults to `muflow.registry.get_all()`.
    task_name : str
        Name for the Celery task. Defaults to "muflow.execute_node".

    Returns
    -------
    celery.Task
        The registered Celery task.

    Example
    -------
    >>> from celery import Celery
    >>> from muflow import registry
    >>>
    >>> # In your Celery worker:
    >>> app = Celery("worker")
    >>> task = create_celery_task(app)
    """
    from muflow import registry
    from muflow.context import TaskContext
    from muflow.executor import ExecutionPayload, execute_task
    from muflow.storage import S3StorageBackend

    if task_registry is None:
        task_registry = registry.get_all()

    @celery_app.task(name="muflow.send_completion")
    def send_completion_task(
        plan_id: str,
        callback_task_name: str,
        callback_queue: str,
    ):
        """Dispatch the completion callback task.

        Called as a chord callback after all plan nodes complete
        successfully.  Sends ``callback_task_name`` to ``callback_queue``
        with ``(plan_id, True, None)`` as arguments.

        Parameters
        ----------
        plan_id : str
            Plan root key.
        callback_task_name : str
            Celery task name to call.
        callback_queue : str
            Queue to send it to.
        """
        celery_app.send_task(
            callback_task_name,
            args=[plan_id, True, None],
            queue=callback_queue,
        )
        _log.info(
            f"Sent completion notification for plan {plan_id} "
            f"to {callback_task_name} on queue {callback_queue}"
        )

    @celery_app.task(name=task_name, bind=True)
    def execute_node_task(
        self,
        node_key: str,
        payload_dict: dict,
        bucket: str,
    ):
        """Execute a single task node.

        Parameters
        ----------
        self : celery.Task
            Bound Celery task instance.
        node_key : str
            Unique key for this node (for logging/tracking).
        payload_dict : dict
            Serialized execution payload.
        bucket : str
            S3 bucket for I/O.

        Returns
        -------
        dict
            Execution result summary.
        """
        payload = ExecutionPayload.from_dict(payload_dict)

        _log.info(
            f"execute_node_task: Starting {payload.task_name} "
            f"(node_key={node_key[:16]}..., task_id={self.request.id})"
        )

        if payload.task_name not in task_registry:
            error_msg = f"Unknown task: {payload.task_name}"
            _log.error(f"execute_node_task: {error_msg}")
            raise ValueError(error_msg)

        # Create storage backends
        storage = S3StorageBackend(payload.storage_prefix, bucket)
        dep_storages = {
            key: S3StorageBackend(prefix, bucket)
            for key, prefix in payload.dependency_prefixes.items()
        }

        ctx = TaskContext(
            storage=storage,
            kwargs=payload.kwargs,
            dependency_storages=dep_storages,
        )

        # Execute
        result = execute_task(
            payload=payload,
            context=ctx,
            get_entry=lambda name: task_registry[name],
        )

        # Write execution metadata to S3
        try:
            ctx.save_json("_execution_result.json", {
                "node_key": node_key,
                "task_id": self.request.id,
                "success": result.success,
                "error_message": result.error_message,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            _log.warning(f"Failed to write _execution_result.json: {e}")

        if result.success:
            _log.info(
                f"execute_node_task: Completed {payload.task_name} "
                f"(node_key={node_key[:16]}...)"
            )
        else:
            _log.error(
                f"execute_node_task: Failed {payload.task_name} "
                f"(node_key={node_key[:16]}...): {result.error_message}"
            )
            raise RuntimeError(result.error_message)

        return {"node_key": node_key}

    return execute_node_task
