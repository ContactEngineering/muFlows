"""Base execution backend protocol and local implementation.

ExecutionBackend is the protocol for submitting workflow plans for execution.
Backends receive an entire WorkflowPlan and orchestrate it using their native
primitives (Celery chord, Step Functions, serial loop, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from muflow.plan import WorkflowPlan

_log = logging.getLogger(__name__)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for workflow plan execution backends.

    An execution backend knows how to:
    - Execute an entire workflow plan (DAG)
    - Orchestrate parallelism using its native primitives
    - Track and report plan state

    Backends handle:
    - DAG traversal and dependency management
    - Parallel execution where possible
    - Node-level execution via execute_workflow()

    Implementations:
    - LocalBackend: Executes serially in-process (for testing/CLI)
    - CeleryBackend: Uses Celery chord/group for parallel execution
    - StepFunctionsBackend: Step Functions orchestration + Lambda execution
    """

    def submit_plan(
        self,
        plan: "WorkflowPlan",
        on_node_start: Optional[Callable[[str], None]] = None,
        on_node_complete: Optional[Callable[[str], None]] = None,
        on_node_failure: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Submit an entire workflow plan for execution.

        The backend orchestrates the DAG using its native primitives.
        For async backends (Celery, Lambda), this returns immediately.
        For sync backends (Local), this blocks until complete.

        Parameters
        ----------
        plan : WorkflowPlan
            Complete workflow plan with all nodes and dependencies.
        on_node_start : callable, optional
            Callback when a node starts: (node_key) -> None
        on_node_complete : callable, optional
            Callback when a node completes: (node_key) -> None
        on_node_failure : callable, optional
            Callback when a node fails: (node_key, error_message) -> None

        Returns
        -------
        str
            Plan execution ID for tracking.
        """
        ...

    def get_plan_state(self, plan_id: str) -> str:
        """Query the state of a plan execution.

        Parameters
        ----------
        plan_id : str
            Plan execution ID returned by submit_plan().

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
            Plan execution ID returned by submit_plan().
        """
        ...


class LocalBackend:
    """Synchronous local execution backend for testing and CLI.

    Executes the entire plan in the current process, serially.
    Nodes are executed in dependency order (topological sort).

    Parameters
    ----------
    base_path : str
        Base directory for workflow storage.
    registry_get : callable, optional
        Function to get workflow entries: (name) -> WorkflowEntry
        Defaults to `muflow.registry.get`.
    progress_reporter : callable, optional
        Function called with (current, total, message) for progress updates.
        Defaults to printing to stdout.

    Example
    -------
    >>> from muflow import WorkflowPlanner, registry
    >>> from muflow.backends import LocalBackend
    >>>
    >>> # Build a plan
    >>> planner = WorkflowPlanner(base_prefix="/tmp/output")
    >>> plan = planner.build_plan("my_workflow", "subject:1", kwargs)
    >>>
    >>> # Execute locally
    >>> backend = LocalBackend("/tmp/output")
    >>> plan_id = backend.submit_plan(plan)
    >>> print(f"Completed: {plan_id}")
    """

    def __init__(
        self,
        base_path: str,
        registry_get: Optional[Callable] = None,
        progress_reporter: Optional[Callable[[int, int, str], None]] = None,
    ):
        """Initialize the local backend.

        Parameters
        ----------
        base_path : str
            Base directory for workflow storage.
        registry_get : callable, optional
            Function to get workflow entries by name.
            Defaults to `muflow.registry.get`.
        progress_reporter : callable, optional
            Function called with (current, total, message) for progress updates.
            Defaults to printing to stdout with right-aligned percentage.
        """
        from muflow import registry

        self.base_path = base_path
        self.registry_get = registry_get or registry.get
        self.progress_reporter = progress_reporter
        self._plan_states: dict[str, str] = {}

    def submit_plan(
        self,
        plan: "WorkflowPlan",
        on_node_start: Optional[Callable[[str], None]] = None,
        on_node_complete: Optional[Callable[[str], None]] = None,
        on_node_failure: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Execute the entire plan synchronously.

        Nodes are executed in dependency order. Execution blocks until
        the entire plan completes or a node fails.

        Parameters
        ----------
        plan : WorkflowPlan
            Complete workflow plan.
        on_node_start : callable, optional
            Callback when a node starts execution.
        on_node_complete : callable, optional
            Callback when a node completes successfully.
        on_node_failure : callable, optional
            Callback when a node fails.

        Returns
        -------
        str
            Plan execution ID (the plan's root_key).

        Raises
        ------
        RuntimeError
            If any node fails during execution.
        """
        from muflow import ExecutionPayload, create_local_context, execute_workflow
        from muflow.planner import get_dependency_access_map

        plan_id = plan.root_key
        self._plan_states[plan_id] = "running"

        # Initialize completed set with cached nodes
        completed = {k for k, n in plan.nodes.items() if n.cached}

        _log.info(f"Executing plan {plan_id} with {len(plan.nodes)} nodes "
                  f"({len(completed)} cached)")

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

                    # Build dependency paths using access keys (not node keys)
                    dependency_paths = get_dependency_access_map(
                        plan, node.key, base_prefix=self.base_path
                    )

                    # Create context
                    ctx = create_local_context(
                        path=node.storage_prefix,
                        kwargs=node.kwargs,
                        dependency_paths=dependency_paths,
                        progress_reporter=self.progress_reporter,
                    )

                    # Build payload
                    payload = ExecutionPayload(
                        workflow_name=node.function,
                        kwargs=node.kwargs,
                        storage_prefix=node.storage_prefix,
                        dependency_prefixes=dependency_paths,
                    )

                    # Execute
                    result = execute_workflow(payload, ctx, self.registry_get)

                    if result.success:
                        completed.add(node.key)
                        _log.debug(f"Node completed: {node.key[:16]}...")
                        if on_node_complete:
                            on_node_complete(node.key)
                    else:
                        self._plan_states[plan_id] = "failure"
                        _log.error(f"Node failed: {node.key[:16]}... - "
                                   f"{result.error_message}")
                        if on_node_failure:
                            on_node_failure(node.key, result.error_message)
                        raise RuntimeError(
                            f"Node {node.function} failed: {result.error_message}\n"
                            f"{result.error_traceback or ''}"
                        )

            self._plan_states[plan_id] = "success"
            _log.info(f"Plan {plan_id} completed successfully")
            return plan_id

        except Exception:
            self._plan_states[plan_id] = "failure"
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
