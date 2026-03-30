"""AWS Lambda execution backend.

This module provides LambdaBackend for executing workflow plans on AWS Lambda.

For simple plans, nodes are invoked serially. For complex parallel execution,
consider using AWS Step Functions to orchestrate Lambda invocations (not yet
implemented in this backend).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from muflow.plan import WorkflowPlan

_log = logging.getLogger(__name__)

try:
    import boto3
except ImportError:
    boto3 = None


class LambdaBackend:
    """Execute workflow plans on AWS Lambda.

    For now, this backend invokes Lambda functions serially for each node.
    For true parallel execution, use AWS Step Functions to orchestrate
    the Lambda invocations.

    Parameters
    ----------
    function_name : str
        Lambda function name to invoke for each node.
    bucket : str
        S3 bucket for workflow data.
    lambda_client : optional
        Boto3 Lambda client. If not provided, one will be created.

    Example
    -------
    >>> from muflow import WorkflowPlanner
    >>> from muflow.backends import LambdaBackend
    >>>
    >>> backend = LambdaBackend(
    ...     function_name="my-workflow-executor",
    ...     bucket="my-bucket",
    ... )
    >>> plan = WorkflowPlanner().build_plan(...)
    >>> plan_id = backend.submit_plan(plan)
    """

    def __init__(
        self,
        function_name: str,
        bucket: str,
        lambda_client=None,
    ):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for LambdaBackend. "
                "Install with: pip install muflow[s3]"
            )

        self._function_name = function_name
        self._bucket = bucket
        self._lambda = lambda_client or boto3.client("lambda")
        self._plan_states: dict[str, str] = {}

    def submit_plan(
        self,
        plan: "WorkflowPlan",
        on_node_complete: Optional[Callable[[str], None]] = None,
        on_node_failure: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Submit a workflow plan for execution.

        Currently executes nodes serially by invoking Lambda synchronously
        for each node in dependency order. For async parallel execution,
        consider using Step Functions.

        Parameters
        ----------
        plan : WorkflowPlan
            Complete workflow plan.
        on_node_complete : callable, optional
            Callback when a node completes.
        on_node_failure : callable, optional
            Callback when a node fails.

        Returns
        -------
        str
            Plan execution ID.
        """
        plan_id = f"lambda-{plan.root_key[:16]}"
        self._plan_states[plan_id] = "running"

        # Initialize completed set with cached nodes
        completed = {k for k, n in plan.nodes.items() if n.cached}

        _log.info(
            f"Executing plan {plan_id} with {len(plan.nodes)} nodes "
            f"({len(completed)} cached)"
        )

        try:
            while not plan.is_complete(completed):
                ready = plan.ready_nodes(completed)

                if not ready:
                    raise RuntimeError(
                        "Deadlock: no nodes ready but plan not complete."
                    )

                for node in ready:
                    _log.debug(f"Invoking Lambda for node: {node.function}")

                    # Build dependency prefixes
                    dependency_prefixes = {
                        dep_key: plan.nodes[dep_key].storage_prefix
                        for dep_key in node.depends_on
                    }

                    # Build event payload
                    event = {
                        "node_key": node.key,
                        "workflow_name": node.function,
                        "kwargs": node.kwargs,
                        "storage_prefix": node.storage_prefix,
                        "dependency_prefixes": dependency_prefixes,
                        "bucket": self._bucket,
                    }

                    # Invoke synchronously
                    response = self._lambda.invoke(
                        FunctionName=self._function_name,
                        InvocationType="RequestResponse",  # Sync
                        Payload=json.dumps(event),
                    )

                    # Check for errors
                    if "FunctionError" in response:
                        payload = json.loads(response["Payload"].read())
                        error_msg = payload.get("errorMessage", "Unknown error")
                        self._plan_states[plan_id] = "failure"
                        if on_node_failure:
                            on_node_failure(node.key, error_msg)
                        raise RuntimeError(f"Lambda failed: {error_msg}")

                    completed.add(node.key)
                    _log.debug(f"Node completed: {node.key[:16]}...")
                    if on_node_complete:
                        on_node_complete(node.key)

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
        """Cancel not supported for Lambda (sync execution).

        Raises
        ------
        NotImplementedError
            Always, since Lambda invocations are synchronous.
        """
        raise NotImplementedError(
            "LambdaBackend executes synchronously and cannot be cancelled. "
            "Consider using Step Functions for cancellable async execution."
        )


def create_lambda_handler(workflow_registry: Optional[dict] = None):
    """Create a Lambda handler function for workflow node execution.

    Parameters
    ----------
    workflow_registry : dict, optional
        Mapping from workflow name to WorkflowEntry.
        Defaults to `muflow.registry.get_all()`.

    Returns
    -------
    callable
        Lambda handler function.

    Example
    -------
    >>> from muflow import registry
    >>> from muflow.backends.aws_lambda import create_lambda_handler
    >>>
    >>> # In your Lambda function module:
    >>> handler = create_lambda_handler()
    """
    from muflow import registry
    from muflow.context import WorkflowContext
    from muflow.executor import ExecutionPayload, execute_workflow
    from muflow.storage import S3StorageBackend

    if workflow_registry is None:
        workflow_registry = registry.get_all()

    def handler(event, context):
        """Lambda handler for workflow node execution.

        Parameters
        ----------
        event : dict
            Event containing:
            - node_key: str
            - workflow_name: str
            - kwargs: dict
            - storage_prefix: str
            - dependency_prefixes: dict
            - bucket: str
        context : LambdaContext
            Lambda context (unused).

        Returns
        -------
        dict
            Execution result.
        """
        workflow_name = event["workflow_name"]
        node_key = event["node_key"]
        bucket = event["bucket"]

        _log.info(f"Lambda handler: Starting {workflow_name} "
                  f"(node_key={node_key[:16]}...)")

        if workflow_name not in workflow_registry:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        payload = ExecutionPayload(
            workflow_name=workflow_name,
            kwargs=event["kwargs"],
            storage_prefix=event["storage_prefix"],
            dependency_prefixes=event.get("dependency_prefixes", {}),
        )

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
