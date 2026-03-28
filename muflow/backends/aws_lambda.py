"""AWS Lambda execution backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from muflow.executor import ExecutionPayload

try:
    import boto3
except ImportError:
    boto3 = None


class LambdaBackend:
    """Execute workflows on AWS Lambda.

    Lambda functions run without Django/database access. They receive
    all necessary information in the event payload and write outputs
    directly to S3. Completion is signaled via Lambda Destinations
    (to SQS) or Step Functions.

    Parameters
    ----------
    function_name : str
        Default Lambda function name to invoke.
    bucket : str
        S3 bucket for workflow data.
    lambda_client : optional
        Boto3 Lambda client. If not provided, one will be created.
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

    def submit(self, analysis_id: int, payload: "ExecutionPayload") -> str:
        """Invoke Lambda function asynchronously.

        Parameters
        ----------
        analysis_id : int
            Database ID of the WorkflowResult.
        payload : ExecutionPayload
            Workflow execution payload.

        Returns
        -------
        str
            AWS Request ID for the invocation.
        """
        # Build Lambda event from ExecutionPayload
        event = {
            "analysis_id": analysis_id,
            "workflow_name": payload.workflow_name,
            "kwargs": payload.kwargs,
            "storage_prefix": payload.storage_prefix,
            "dependency_prefixes": payload.dependency_prefixes,
            "allowed_outputs": list(payload.allowed_outputs),
            "bucket": self._bucket,
        }

        # Invoke asynchronously
        response = self._lambda.invoke(
            FunctionName=self._function_name,
            InvocationType="Event",  # Async
            Payload=json.dumps(event),
        )

        # Return request ID as task ID
        return response["ResponseMetadata"]["RequestId"]

    def cancel(self, task_id: str) -> None:
        """Cancel not supported for Lambda.

        Lambda functions cannot be cancelled once invoked. The function
        will run to completion or timeout.
        """
        raise NotImplementedError(
            "Lambda invocations cannot be cancelled. "
            "Consider using Step Functions for cancellation support."
        )

    def get_state(self, task_id: str) -> str:
        """State tracking not supported for async Lambda.

        For async Lambda invocations, state must be tracked via:
        - Lambda Destinations (on success/failure)
        - Step Functions
        - Custom status updates to S3/DynamoDB

        Returns
        -------
        str
            Always returns "pending" since we can't query Lambda state.
        """
        # Lambda async invocations don't provide state queries
        # State must be tracked via destinations or callbacks
        return "pending"


def create_lambda_handler(workflow_registry: dict):
    """Create a Lambda handler function for workflow execution.

    This is a factory function that creates a Lambda handler configured
    with a registry of available workflow implementations.

    Parameters
    ----------
    workflow_registry : dict
        Mapping from workflow name to implementation class.

    Returns
    -------
    callable
        Lambda handler function.

    Example
    -------
    >>> from myworkflows import GPRWorkflow, GPCWorkflow
    >>> handler = create_lambda_handler({
    ...     "sds_ml.v3.gpr.training": GPRWorkflow,
    ...     "sds_ml.v3.gpc.training": GPCWorkflow,
    ... })
    """
    from muflow.context import S3WorkflowContext
    from muflow.executor import ExecutionPayload, execute_workflow

    def handler(event, context):
        """Lambda handler for workflow execution.

        Event structure (matches ExecutionPayload + Lambda-specific fields):
        {
            "analysis_id": 123,
            "workflow_name": "sds_ml.v3.gpr.training",
            "kwargs": {...},
            "storage_prefix": "data-lake/results/...",
            "dependency_prefixes": {"dep1": "data-lake/..."},
            "allowed_outputs": ["result.json", "model.nc"],
            "bucket": "my-bucket"
        }
        """
        workflow_name = event["workflow_name"]

        if workflow_name not in workflow_registry:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Reconstruct ExecutionPayload from event
        payload = ExecutionPayload(
            workflow_name=workflow_name,
            kwargs=event["kwargs"],
            storage_prefix=event["storage_prefix"],
            dependency_prefixes=event.get("dependency_prefixes", {}),
            allowed_outputs=set(event.get("allowed_outputs", [])),
        )

        # Create S3 context
        ctx = S3WorkflowContext(
            storage_prefix=payload.storage_prefix,
            kwargs=payload.kwargs,
            dependency_prefixes=payload.dependency_prefixes,
            bucket=event["bucket"],
            allowed_outputs=payload.allowed_outputs,
        )

        # Execute using the pure execution function
        result = execute_workflow(
            payload=payload,
            context=ctx,
            get_implementation=lambda name: workflow_registry[name],
        )

        if not result.success:
            # Re-raise the error for Lambda to handle
            raise RuntimeError(result.error_message)

        return {
            "status": "success",
            "analysis_id": event["analysis_id"],
            "files_written": result.files_written,
        }

    return handler
