"""AWS Step Functions execution backend.

Replaces the serial LambdaBackend with a proper async orchestrator.
Step Functions is the AWS equivalent of a Celery canvas:

  Celery                    Step Functions
  ─────────────────────     ──────────────────────────────────
  group(t1, t2, t3)      →  Parallel state (3 branches)
  chord(group, next)     →  Parallel state → Next state
  apply_async()          →  StartExecution()  (returns immediately)
  AsyncResult.state      →  DescribeExecution()
  revoke()               →  StopExecution()

Architecture
------------
submit_plan() translates the WorkflowPlan DAG into an ASL state machine:

1. Topologically sort nodes into levels (same algorithm as CeleryBackend).
2. A single-node level becomes a Task state; a multi-node level becomes a
   Parallel state whose branches run concurrently.
3. Levels are linked in sequence via Next pointers.
4. The ASL is pushed to a new (or updated) Step Functions state machine.
5. StartExecution() is called; the execution ARN is returned immediately.
6. AWS orchestrates everything from there — the caller is free.

Each node is still executed by a Lambda function (create_lambda_handler).
Step Functions controls *when* each Lambda is invoked and handles retries.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from muflow.plan import WorkflowNode, WorkflowPlan

_log = logging.getLogger(__name__)

try:
    import boto3
except ImportError:
    boto3 = None


class StepFunctionsBackend:
    """Execute workflow plans via AWS Step Functions + Lambda.

    Translates a WorkflowPlan DAG into an ASL state machine, registers it
    with Step Functions, and starts an execution.  Returns immediately;
    AWS owns the orchestration from that point.

    Independent nodes (same dependency depth) run in parallel via a Parallel
    state.  Dependent levels are sequenced so each level starts only after
    every node in the previous level has completed successfully.

    Parameters
    ----------
    function_arn : str
        ARN of the Lambda function that executes individual workflow nodes.
        Create it with :func:`create_lambda_handler`.
    bucket : str
        S3 bucket for workflow data.
    role_arn : str
        IAM role ARN that Step Functions uses to invoke Lambda.
        The role must have ``lambda:InvokeFunction`` permission.
    base_prefix : str
        S3 key prefix that was passed to ``WorkflowPlanner`` when the plan
        was built.  Used to recompute dependency storage prefixes.
        Default: ``"muflow"``.
    state_machine_prefix : str
        Short string prepended to auto-generated state machine names.
        Default: ``"muflow"``.
    sfn_client : optional
        Pre-built boto3 Step Functions client.  Created automatically if
        omitted (useful for injecting mocks in tests).

    Example
    -------
    >>> from muflow import WorkflowPlanner
    >>> from muflow.backends import StepFunctionsBackend
    >>>
    >>> backend = StepFunctionsBackend(
    ...     function_arn="arn:aws:lambda:us-east-1:123456789:function:muflow-worker",
    ...     bucket="my-workflow-bucket",
    ...     role_arn="arn:aws:iam::123456789:role/StepFunctionsLambdaRole",
    ... )
    >>> plan = WorkflowPlanner(base_prefix="muflow").build_plan(
    ...     "myapp.analysis", "dataset:42", {"param": "value"}
    ... )
    >>> execution_arn = backend.submit_plan(plan)
    >>> # Returns immediately.  Poll later:
    >>> state = backend.get_plan_state(execution_arn)  # "running" | "success" | …
    """

    def __init__(
        self,
        function_arn: str,
        bucket: str,
        role_arn: str,
        base_prefix: str = "muflow",
        state_machine_prefix: str = "muflow",
        sfn_client=None,
    ):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for StepFunctionsBackend. "
                "Install with: pip install muflow[s3]"
            )
        self._function_arn = function_arn
        self._bucket = bucket
        self._role_arn = role_arn
        self._base_prefix = base_prefix
        self._state_machine_prefix = state_machine_prefix
        self._sfn = sfn_client or boto3.client("stepfunctions")

    # ── ExecutionBackend protocol ──────────────────────────────────────────

    def submit_plan(
        self,
        plan: "WorkflowPlan",
        on_node_start: Optional[Callable[[str], None]] = None,
        on_node_complete: Optional[Callable[[str], None]] = None,
        on_node_failure: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Translate the plan to ASL, create a state machine, and start execution.

        Returns immediately with the execution ARN.  AWS Step Functions owns
        the orchestration from this point.

        Node-level callbacks are not supported because execution is fully
        async.  Use CloudWatch EventBridge rules on Step Functions state-change
        events for equivalent notification.

        Parameters
        ----------
        plan : WorkflowPlan
            Complete workflow plan.
        on_node_start, on_node_complete, on_node_failure : callable, optional
            Ignored (async execution).  A warning is logged if any are passed.

        Returns
        -------
        str
            Step Functions execution ARN, or the sentinel
            ``"cached-{root_key}"`` when every node is already cached.
        """
        if any(cb is not None for cb in (on_node_start, on_node_complete, on_node_failure)):
            _log.warning(
                "StepFunctionsBackend: node callbacks are not supported "
                "(execution is fully async).  Use CloudWatch EventBridge "
                "on Step Functions state-change events instead."
            )

        levels = self._compute_levels(plan)
        asl = self._build_asl(levels, plan)

        if asl is None:
            # Every node was already cached — nothing to run.
            _log.info(f"Plan {plan.root_key[:24]}...: all nodes cached")
            return f"cached-{plan.root_key}"

        sm_arn = self._ensure_state_machine(self._state_machine_name(plan.root_key), asl)

        execution_name = f"exec-{uuid.uuid4().hex[:24]}"
        resp = self._sfn.start_execution(
            stateMachineArn=sm_arn,
            name=execution_name,
            input="{}",
        )
        execution_arn = resp["executionArn"]

        _log.info(
            f"Started Step Functions execution: {execution_arn} "
            f"({len(plan.nodes)} nodes)"
        )
        return execution_arn

    def get_plan_state(self, execution_arn: str) -> str:
        """Query the current state of a plan execution.

        Parameters
        ----------
        execution_arn : str
            ARN returned by :meth:`submit_plan`, or the ``"cached-…"`` sentinel.

        Returns
        -------
        str
            One of: ``"pending"``, ``"running"``, ``"success"``, ``"failure"``.
        """
        if execution_arn.startswith("cached-"):
            return "success"

        resp = self._sfn.describe_execution(executionArn=execution_arn)
        return {
            "RUNNING": "running",
            "SUCCEEDED": "success",
            "FAILED": "failure",
            "TIMED_OUT": "failure",
            "ABORTED": "failure",
        }.get(resp["status"], "pending")

    def cancel_plan(self, execution_arn: str) -> None:
        """Stop a running execution.

        Parameters
        ----------
        execution_arn : str
            ARN returned by :meth:`submit_plan`.
        """
        if execution_arn.startswith("cached-"):
            return
        self._sfn.stop_execution(
            executionArn=execution_arn,
            cause="Cancelled by muflow StepFunctionsBackend",
        )
        _log.info(f"Stopped execution: {execution_arn}")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _state_machine_name(self, root_key: str) -> str:
        """Derive a valid Step Functions state machine name from a root key.

        AWS state machine names: 1–80 chars, ``[A-Za-z0-9_-]`` only.
        ``root_key`` looks like ``"muflow/my.workflow/a1b2c3d4e5f6g7h8"``; we
        use the hash suffix for uniqueness.
        """
        suffix = root_key.rsplit("/", 1)[-1] if "/" in root_key else root_key
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", suffix)
        return f"{self._state_machine_prefix}-{sanitized}"[:80]

    def _ensure_state_machine(self, name: str, asl: dict) -> str:
        """Create the named state machine, or update its definition if it exists.

        Parameters
        ----------
        name : str
            State machine name (already validated for AWS naming rules).
        asl : dict
            ASL definition.

        Returns
        -------
        str
            State machine ARN.
        """
        definition = json.dumps(asl)
        try:
            resp = self._sfn.create_state_machine(
                name=name,
                definition=definition,
                roleArn=self._role_arn,
                type="STANDARD",
            )
            arn = resp["stateMachineArn"]
            _log.debug(f"Created state machine: {arn}")
            return arn
        except self._sfn.exceptions.StateMachineAlreadyExists:
            pass

        # The state machine already exists — find its ARN and update it.
        paginator = self._sfn.get_paginator("list_state_machines")
        for page in paginator.paginate():
            for sm in page["stateMachines"]:
                if sm["name"] == name:
                    arn = sm["stateMachineArn"]
                    self._sfn.update_state_machine(
                        stateMachineArn=arn,
                        definition=definition,
                    )
                    _log.debug(f"Updated state machine: {arn}")
                    return arn

        raise RuntimeError(  # should never happen
            f"State machine '{name}' exists but could not be located."
        )

    def _compute_levels(self, plan: "WorkflowPlan") -> list[list["WorkflowNode"]]:
        """Group non-cached nodes by execution level (topological sort).

        Level 0 contains nodes whose dependencies are all already cached or
        have no dependencies.  Level N contains nodes whose dependencies are
        all in levels < N.  Cached nodes are excluded from all levels.

        Raises ``ValueError`` on circular dependencies.
        """
        levels: list[list["WorkflowNode"]] = []
        completed = {k for k, n in plan.nodes.items() if n.cached}
        remaining = set(plan.nodes.keys()) - completed

        while remaining:
            ready = [
                plan.nodes[key]
                for key in remaining
                if all(d in completed for d in plan.nodes[key].depends_on)
            ]
            if not ready:
                raise ValueError(
                    f"Circular dependency detected. Unresolvable nodes: {remaining}"
                )
            levels.append(ready)
            completed.update(n.key for n in ready)
            remaining -= {n.key for n in ready}

        return levels

    def _build_asl(
        self,
        levels: list[list["WorkflowNode"]],
        plan: "WorkflowPlan",
    ) -> Optional[dict]:
        """Build an ASL state machine definition from execution levels.

        A single-node level becomes a Task state.  A multi-node level becomes
        a Parallel state whose branches all run concurrently.  Levels are
        linked via ``Next`` pointers; the last level carries ``"End": true``.

        Returns ``None`` when ``levels`` is empty (all nodes cached).
        """
        if not levels:
            return None

        state_names = [f"Level{i}" for i in range(len(levels))]
        states = {}

        for i, (state_name, nodes) in enumerate(zip(state_names, levels)):
            if len(nodes) == 1:
                state = self._task_state(nodes[0], plan)
            else:
                state = self._parallel_state(nodes, plan)

            if i < len(levels) - 1:
                state["Next"] = state_names[i + 1]
            else:
                state["End"] = True

            states[state_name] = state

        return {
            "Comment": f"muFlow plan: {plan.root_key}",
            "StartAt": state_names[0],
            "States": states,
        }

    def _task_state(self, node: "WorkflowNode", plan: "WorkflowPlan") -> dict:
        """ASL Task state that invokes Lambda for a single node."""
        return {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": self._function_arn,
                "Payload": self._node_payload(node, plan),
            },
            "ResultPath": None,
            "Retry": [
                {
                    "ErrorEquals": [
                        "Lambda.ServiceException",
                        "Lambda.AWSLambdaException",
                        "Lambda.SdkClientException",
                        "Lambda.TooManyRequestsException",
                    ],
                    "IntervalSeconds": 2,
                    "MaxAttempts": 3,
                    "BackoffRate": 2.0,
                }
            ],
        }

    def _parallel_state(
        self, nodes: list["WorkflowNode"], plan: "WorkflowPlan"
    ) -> dict:
        """ASL Parallel state that runs multiple nodes concurrently."""
        branches = []
        for node in nodes:
            task = self._task_state(node, plan)
            task["End"] = True
            branches.append({
                "StartAt": "Execute",
                "States": {"Execute": task},
            })
        return {
            "Type": "Parallel",
            "Branches": branches,
            "ResultPath": None,
        }

    def _node_payload(self, node: "WorkflowNode", plan: "WorkflowPlan") -> dict:
        """Build the Lambda event payload for a node.

        Dependency prefixes are keyed by their *access keys* (e.g.
        ``"surface_0"``) so that workflow code can call
        ``ctx.dependency("surface_0")`` correctly.
        """
        from muflow.planner import get_dependency_access_map

        dependency_prefixes = get_dependency_access_map(
            plan, node.key, base_prefix=self._base_prefix
        )
        return {
            "workflow_name": node.function,
            "kwargs": node.kwargs,
            "storage_prefix": node.storage_prefix,
            "dependency_prefixes": dependency_prefixes,
            "bucket": self._bucket,
            "node_key": node.key,
        }


# ── Lambda node executor ───────────────────────────────────────────────────


def create_lambda_handler(workflow_registry: Optional[dict] = None):
    """Create a Lambda handler function for workflow node execution.

    The handler is the compute half of the Step Functions + Lambda
    architecture: Step Functions decides *when* to call it; this function
    does the actual work for a single node.

    Parameters
    ----------
    workflow_registry : dict, optional
        Mapping ``{workflow_name: WorkflowEntry}``.
        Defaults to ``muflow.registry.get_all()`` at call time.

    Returns
    -------
    callable
        A Lambda handler with signature ``handler(event, context) -> dict``.

    Example
    -------
    In your Lambda function module::

        from muflow.backends.step_functions import create_lambda_handler

        # Register all workflows before creating the handler
        import myapp.workflows  # noqa: F401 (side-effect: registers workflows)

        handler = create_lambda_handler()
    """
    from muflow import registry
    from muflow.context import WorkflowContext
    from muflow.executor import ExecutionPayload, execute_workflow
    from muflow.storage import S3StorageBackend

    if workflow_registry is None:
        workflow_registry = registry.get_all()

    def handler(event: dict, context) -> dict:
        """Execute a single workflow node.

        Expected event keys
        -------------------
        workflow_name : str
        kwargs : dict
        storage_prefix : str
            S3 key prefix for this node's outputs.
        dependency_prefixes : dict[str, str]
            Access-key → S3 key prefix for each declared dependency.
        bucket : str
        node_key : str
            Opaque node identifier (used for logging).

        Returns
        -------
        dict
            ``{"status": "success", "node_key": …, "files_written": […]}``

        Raises
        ------
        ValueError
            If the workflow name is not in the registry.
        RuntimeError
            If workflow execution fails.
        """
        workflow_name = event["workflow_name"]
        node_key = event["node_key"]
        bucket = event["bucket"]

        _log.info(f"Lambda handler: {workflow_name} (node={node_key[:16]}...)")

        if workflow_name not in workflow_registry:
            raise ValueError(f"Unknown workflow: {workflow_name!r}")

        payload = ExecutionPayload(
            workflow_name=workflow_name,
            kwargs=event["kwargs"],
            storage_prefix=event["storage_prefix"],
            dependency_prefixes=event.get("dependency_prefixes", {}),
        )

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

        result = execute_workflow(
            payload=payload,
            context=ctx,
            get_entry=lambda name: workflow_registry[name],
        )

        if not result.success:
            raise RuntimeError(result.error_message)

        return {
            "status": "success",
            "node_key": node_key,
            "files_written": result.files_written,
        }

    return handler
