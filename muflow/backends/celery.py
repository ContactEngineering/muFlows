"""Celery execution backend with parallel DAG orchestration.

This module provides a CeleryBackend that executes workflow plans using
Celery's chord and group primitives for parallel execution.

Architecture
------------
- submit_plan() converts the WorkflowPlan DAG into Celery chord/group structures
- Nodes at the same "level" (same dependency depth) run in parallel via group()
- Levels are chained together via chord() so each level waits for the previous
- Workers execute individual nodes via execute_workflow()

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
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from muflow.plan import WorkflowNode, WorkflowPlan

_log = logging.getLogger(__name__)


class CeleryBackend:
    """Celery backend with parallel DAG orchestration.

    Converts WorkflowPlan into Celery chord/group structures for efficient
    parallel execution. Nodes at the same dependency level run in parallel.

    Parameters
    ----------
    celery_app
        Celery application instance.
    bucket : str
        S3 bucket for workflow I/O.
    base_prefix : str
        S3 key prefix that was passed to ``WorkflowPlanner`` when the plan
        was built.  Used to recompute dependency storage prefixes keyed by
        their access names.  Default: ``"muflow"``.
    task_name : str
        Name of the Celery task for node execution.
        Defaults to "muflow.execute_node".

    Example
    -------
    >>> from celery import Celery
    >>> from muflow import WorkflowPlanner
    >>> from muflow.backends import CeleryBackend
    >>>
    >>> app = Celery("myapp")
    >>> backend = CeleryBackend(app, bucket="my-bucket", base_prefix="muflow")
    >>>
    >>> plan = WorkflowPlanner(base_prefix="muflow").build_plan(...)
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
        plan: "WorkflowPlan",
        on_node_complete: Optional[Callable[[str], None]] = None,
        on_node_failure: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Submit a workflow plan for parallel execution.

        Converts the DAG into Celery chord/group structures:
        - Nodes with no dependencies (level 0) run first in parallel
        - Nodes at level N run after all level N-1 nodes complete
        - Within each level, nodes run in parallel

        Parameters
        ----------
        plan : WorkflowPlan
            Complete workflow plan.
        on_node_complete : callable, optional
            Not directly supported - use Celery signals or callbacks task.
        on_node_failure : callable, optional
            Not directly supported - use Celery signals or callbacks task.

        Returns
        -------
        str
            Celery task ID for the outermost chord/group.
        """
        # Compute execution levels (topological sort by depth)
        levels = self._compute_levels(plan)

        _log.info(
            f"Submitting plan {plan.root_key[:16]}... with {len(plan.nodes)} nodes "
            f"in {len(levels)} levels"
        )

        # Build Celery workflow from levels
        celery_workflow = self._build_celery_workflow(levels, plan)

        if celery_workflow is None:
            # All nodes are cached - nothing to execute
            _log.info(f"Plan {plan.root_key[:16]}... - all nodes cached")
            return f"cached-{plan.root_key}"

        # Submit the workflow
        result = celery_workflow.apply_async()
        self._plan_results[result.id] = result

        _log.info(f"Submitted plan as Celery task {result.id}")
        return result.id

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
        if plan_id.startswith("cached-"):
            return "success"

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
        if plan_id.startswith("cached-"):
            return  # Nothing to cancel

        self._app.control.revoke(plan_id, terminate=True)
        _log.info(f"Cancelled plan {plan_id}")

    def _compute_levels(self, plan: "WorkflowPlan") -> list[list["WorkflowNode"]]:
        """Group nodes by execution level (topological sort).

        Level 0: nodes with no dependencies (leaf nodes)
        Level N: nodes whose dependencies are all in levels < N

        Parameters
        ----------
        plan : WorkflowPlan
            The workflow plan.

        Returns
        -------
        list[list[WorkflowNode]]
            Nodes grouped by execution level.
        """
        levels: list[list["WorkflowNode"]] = []
        remaining = set(plan.nodes.keys())
        completed = {k for k, n in plan.nodes.items() if n.cached}

        while remaining - completed:
            # Find nodes whose deps are all complete
            ready = []
            for key in remaining - completed:
                node = plan.nodes[key]
                deps_satisfied = all(
                    d in completed for d in node.depends_on
                )
                if deps_satisfied:
                    ready.append(node)

            if not ready:
                # Check for circular dependency
                raise ValueError(
                    f"Circular dependency detected. Remaining nodes: "
                    f"{remaining - completed}"
                )

            levels.append(ready)
            completed.update(n.key for n in ready)

        return levels

    def _build_celery_workflow(
        self,
        levels: list[list["WorkflowNode"]],
        plan: "WorkflowPlan",
    ):
        """Convert execution levels into Celery chord/chain structure.

        Parameters
        ----------
        levels : list[list[WorkflowNode]]
            Nodes grouped by execution level.
        plan : WorkflowPlan
            The workflow plan (for dependency prefixes).

        Returns
        -------
        celery.canvas.Signature or None
            Celery workflow signature, or None if all nodes are cached.
        """
        from celery import chord, group

        # Build groups for each level (excluding cached nodes)
        celery_groups = []
        for level in levels:
            tasks = []
            for node in level:
                if node.cached:
                    continue
                task = self._make_node_task(node, plan)
                tasks.append(task)

            if tasks:
                celery_groups.append(group(tasks))

        if not celery_groups:
            return None

        if len(celery_groups) == 1:
            return celery_groups[0]

        # Chain levels together using chord
        # chord(group_a, group_b) means: run group_a, when ALL complete, run group_b
        # For multiple levels: chord(g0, chord(g1, chord(g2, g3)))
        # Or simpler: chain them with a dummy aggregator

        # Build from the end backwards
        result = celery_groups[-1]
        for grp in reversed(celery_groups[:-1]):
            # chord(grp, result) - grp runs first, then result
            result = chord(grp, result)

        return result

    def _make_node_task(self, node: "WorkflowNode", plan: "WorkflowPlan"):
        """Create a Celery task signature for a node.

        Parameters
        ----------
        node : WorkflowNode
            The node to create a task for.
        plan : WorkflowPlan
            The workflow plan (for dependency prefixes).

        Returns
        -------
        celery.canvas.Signature
            Celery task signature.
        """
        # Build dependency prefixes keyed by access name (e.g. "surface_0"),
        # not by node key, so workflows can call ctx.dependency("surface_0").
        from muflow.planner import get_dependency_access_map
        dependency_prefixes = get_dependency_access_map(
            plan, node.key, base_prefix=self._base_prefix
        )

        # Build payload dict
        payload_dict = {
            "workflow_name": node.function,
            "kwargs": node.kwargs,
            "storage_prefix": node.storage_prefix,
            "dependency_prefixes": dependency_prefixes,
        }

        # Get queue from node or use default
        queue = getattr(node, 'queue', None) or "default"

        # Create task signature
        return self._app.signature(
            self._task_name,
            args=[node.key, payload_dict, self._bucket],
            queue=queue,
        )


def create_celery_task(
    celery_app,
    workflow_registry: Optional[dict] = None,
    task_name: str = "muflow.execute_node",
):
    """Create a Celery task for executing workflow nodes.

    This is a factory function that creates a Celery task configured
    with a registry of available workflow implementations. The task
    is called by CeleryBackend for each node in the plan.

    Parameters
    ----------
    celery_app
        Celery application instance.
    workflow_registry : dict, optional
        Mapping from workflow name to WorkflowEntry (or legacy class).
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
    from muflow.context import WorkflowContext
    from muflow.executor import ExecutionPayload, execute_workflow
    from muflow.storage import S3StorageBackend

    if workflow_registry is None:
        workflow_registry = registry.get_all()

    @celery_app.task(name=task_name, bind=True)
    def execute_node_task(
        self,
        node_key: str,
        payload_dict: dict,
        bucket: str,
    ):
        """Execute a single workflow node.

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
            f"execute_node_task: Starting {payload.workflow_name} "
            f"(node_key={node_key[:16]}..., task_id={self.request.id})"
        )

        if payload.workflow_name not in workflow_registry:
            error_msg = f"Unknown workflow: {payload.workflow_name}"
            _log.error(f"execute_node_task: {error_msg}")
            raise ValueError(error_msg)

        # Create storage backends
        storage = S3StorageBackend(payload.storage_prefix, bucket)
        dep_storages = {
            key: S3StorageBackend(prefix, bucket)
            for key, prefix in payload.dependency_prefixes.items()
        }

        ctx = WorkflowContext(
            storage=storage,
            kwargs=payload.kwargs,
            dependency_storages=dep_storages,
        )

        # Execute
        result = execute_workflow(
            payload=payload,
            context=ctx,
            get_entry=lambda name: workflow_registry[name],
        )

        # Write execution metadata to S3
        try:
            ctx.save_json("_execution_result.json", {
                "node_key": node_key,
                "task_id": self.request.id,
                "success": result.success,
                "error_message": result.error_message,
                "files_written": result.files_written,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            _log.warning(f"Failed to write _execution_result.json: {e}")

        if result.success:
            _log.info(
                f"execute_node_task: Completed {payload.workflow_name} "
                f"(node_key={node_key[:16]}...)"
            )
        else:
            _log.error(
                f"execute_node_task: Failed {payload.workflow_name} "
                f"(node_key={node_key[:16]}...): {result.error_message}"
            )
            raise RuntimeError(result.error_message)

        return {
            "node_key": node_key,
            "success": result.success,
            "files_written": result.files_written,
        }

    return execute_node_task
