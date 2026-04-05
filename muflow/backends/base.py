"""Base execution backend protocol and local implementation.

ExecutionBackend is the protocol for submitting task plans for execution.
Backends receive an entire TaskPlan and orchestrate it using their native
primitives (Celery chord, Step Functions, serial loop, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from muflow.backends.callbacks import CompletionCallback
    from muflow.backends.handle import PlanHandle
    from muflow.plan import TaskPlan

_log = logging.getLogger(__name__)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for task plan execution backends.

    An execution backend knows how to:
    - Execute an entire task plan (DAG)
    - Orchestrate parallelism using its native primitives
    - Track and report plan state

    Backends handle:
    - DAG traversal and dependency management
    - Parallel execution where possible
    - Node-level execution via execute_task()

    Implementations:
    - LocalBackend: Executes serially in-process (for testing/CLI)
    - CeleryBackend: Uses Celery chord/group for parallel execution
    - StepFunctionsBackend: Step Functions orchestration + Lambda execution
    """

    def submit_plan(
        self,
        plan: "TaskPlan",
        completion_callback: Optional["CompletionCallback"] = None,
    ) -> "PlanHandle":
        """Submit an entire task plan for execution.

        The backend orchestrates the DAG using its native primitives.
        For async backends (Celery, Lambda), this returns immediately.
        For sync backends (Local), this blocks until complete.

        Parameters
        ----------
        plan : TaskPlan
            Complete task plan with all nodes and dependencies.
        completion_callback : CompletionCallback, optional
            Called when the plan finishes (success or failure).
            Receives (plan_id, success, error).

        Returns
        -------
        PlanHandle
            Serializable handle for querying state and cancelling.
        """
        ...

    def get_plan_state(self, plan_id: str) -> str:
        """Query the state of a plan execution.

        Parameters
        ----------
        plan_id : str
            Plan execution ID (from PlanHandle.plan_id).

        Returns
        -------
        str
            One of: "pending", "running", "success", "failure"
        """
        ...

    def cancel_plan(self, plan_id: str) -> None:
        """Cancel a running plan execution.

        Parameters
        ----------
        plan_id : str
            Plan execution ID (from PlanHandle.plan_id).
        """
        ...


class LocalBackend:
    """Synchronous local execution backend for testing and CLI.

    Executes the entire plan in the current process, serially.
    Nodes are executed in dependency order (topological sort).

    Parameters
    ----------
    base_path : str
        Base directory for task storage.
    registry_get : callable, optional
        Function to get task entries: (name) -> TaskEntry
        Defaults to `muflow.registry.get`.
    progress_reporter : callable, optional
        Function called with (current, total, message) for progress updates.
        Defaults to printing to stdout.

    Example
    -------
    >>> from muflow import Pipeline, Step
    >>> from muflow.backends import LocalBackend
    >>>
    >>> # Build a plan from a pipeline
    >>> plan = my_pipeline.build_plan("subject:1", kwargs, base_prefix="/tmp/output")
    >>>
    >>> # Execute locally
    >>> backend = LocalBackend("/tmp/output")
    >>> handle = backend.submit_plan(plan)
    >>> print(f"Completed: {handle.plan_id}")
    """

    def __init__(
        self,
        base_path: str,
        registry_get: Optional[Callable] = None,
        progress_reporter: Optional[Callable[[int, int, str], None]] = None,
    ):
        from muflow import registry

        self.base_path = base_path
        self.registry_get = registry_get or registry.get
        self.progress_reporter = progress_reporter
        self._plan_states: dict[str, str] = {}

    def submit_plan(
        self,
        plan: "TaskPlan",
        completion_callback: Optional["CompletionCallback"] = None,
        on_node_start: Optional[Callable[[str], None]] = None,
        on_node_complete: Optional[Callable[[str], None]] = None,
        on_node_failure: Optional[Callable[[str, str], None]] = None,
    ) -> "PlanHandle":
        """Execute the entire plan synchronously.

        Nodes are executed in dependency order. Execution blocks until
        the entire plan completes or a node fails.

        Parameters
        ----------
        plan : TaskPlan
            Complete task plan.
        completion_callback : CompletionCallback, optional
            Called with (plan_id, success, error) when the plan finishes.
        on_node_start : callable, optional
            Called with (node_key) when a node starts. Local-only.
        on_node_complete : callable, optional
            Called with (node_key) when a node completes. Local-only.
        on_node_failure : callable, optional
            Called with (node_key, error_message) when a node fails. Local-only.

        Returns
        -------
        PlanHandle
            Handle with backend="local". get_state() always returns "success".

        Raises
        ------
        RuntimeError
            If any node fails during execution.
        """
        from muflow import ExecutionPayload, create_local_context, execute_task
        from muflow.backends.handle import PlanHandle

        plan_id = plan.root_key
        self._plan_states[plan_id] = "running"

        completed: set[str] = set()

        _log.info(f"Executing plan {plan_id} with {len(plan.nodes)} nodes")

        try:
            while not plan.is_complete(completed):
                ready = plan.ready_nodes(completed)

                if not ready:
                    raise RuntimeError(
                        "Deadlock: no nodes ready but plan not complete. "
                        "This indicates a circular dependency or missing node."
                    )

                for node in ready:
                    _log.debug(f"Executing node: {node.function} ({node.key[:16]}...)")

                    if on_node_start:
                        on_node_start(node.key)

                    dependency_paths = node.dependency_access_map

                    ctx = create_local_context(
                        path=node.storage_prefix,
                        kwargs=node.kwargs,
                        dependency_paths=dependency_paths,
                        progress_reporter=self.progress_reporter,
                    )

                    payload = ExecutionPayload(
                        task_name=node.function,
                        kwargs=node.kwargs,
                        storage_prefix=node.storage_prefix,
                        dependency_prefixes=dependency_paths,
                    )

                    result = execute_task(payload, ctx, self.registry_get)

                    if result.success:
                        completed.add(node.key)
                        _log.debug(f"Node completed: {node.key[:16]}...")
                        if on_node_complete:
                            on_node_complete(node.key)
                    else:
                        self._plan_states[plan_id] = "failure"
                        _log.error(
                            f"Node failed: {node.key[:16]}... - {result.error_message}"
                        )
                        if on_node_failure:
                            on_node_failure(node.key, result.error_message)
                        raise RuntimeError(
                            f"Node {node.function} failed: {result.error_message}\n"
                            f"{result.error_traceback or ''}"
                        )

            self._plan_states[plan_id] = "success"
            _log.info(f"Plan {plan_id} completed successfully")

            if completion_callback:
                completion_callback.notify(plan_id, success=True)

            return PlanHandle(
                backend="local",
                plan_id=plan_id,
                node_prefixes={k: n.storage_prefix for k, n in plan.nodes.items()},
                storage_type="local",
                storage_config={},
            )

        except Exception as exc:
            self._plan_states[plan_id] = "failure"
            if completion_callback:
                error_msg = str(exc)
                completion_callback.notify(plan_id, success=False, error=error_msg)
            raise

    def get_plan_state(self, plan_id: str) -> str:
        """Get the state of a plan execution.

        Parameters
        ----------
        plan_id : str
            Plan execution ID.

        Returns
        -------
        str
            One of: "pending", "running", "success", "failure"
        """
        return self._plan_states.get(plan_id, "pending")

    def cancel_plan(self, plan_id: str) -> None:
        """Cancel not supported for synchronous execution.

        Raises
        ------
        NotImplementedError
            Always, since local execution is synchronous.
        """
        raise NotImplementedError(
            "LocalBackend executes synchronously and cannot be cancelled"
        )
