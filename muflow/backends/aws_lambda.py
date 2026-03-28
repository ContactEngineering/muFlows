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
        event = {
            "analysis_id": analysis_id,
            "workflow_name": payload.workflow_name,
            "kwargs": payload.kwargs,
            "storage_prefix": payload.storage_prefix,
            "dependency_prefixes": payload.dependency_prefixes,
            "bucket": self._bucket,
        }

        response = self._lambda.invoke(
            FunctionName=self._function_name,
            InvocationType="Event",  # Async
            Payload=json.dumps(event),
        )

        return response["ResponseMetadata"]["RequestId"]

    def cancel(self, task_id: str) -> None:
        """Cancel not supported for Lambda."""
        raise NotImplementedError(
            "Lambda invocations cannot be cancelled. "
            "Consider using Step Functions for cancellation support."
        )

    def get_state(self, task_id: str) -> str:
        """State tracking not supported for async Lambda.

        Returns
        -------
        str
            Always returns "pending" since we can't query Lambda state.
        """
        return "pending"


def create_lambda_handler(workflow_registry: dict):
    """Create a Lambda handler function for workflow execution.

    Parameters
    ----------
    workflow_registry : dict
        Mapping from workflow name to ``WorkflowEntry`` (or legacy class).

    Returns
    -------
    callable
        Lambda handler function.
    """
    from muflow.context import S3WorkflowContext
    from muflow.executor import ExecutionPayload, execute_workflow

    def handler(event, context):
        """Lambda handler for workflow execution."""
        workflow_name = event["workflow_name"]

        if workflow_name not in workflow_registry:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        payload = ExecutionPayload(
            workflow_name=workflow_name,
            kwargs=event["kwargs"],
            storage_prefix=event["storage_prefix"],
            dependency_prefixes=event.get("dependency_prefixes", {}),
        )

        ctx = S3WorkflowContext(
            storage_prefix=payload.storage_prefix,
            kwargs=payload.kwargs,
            dependency_prefixes=payload.dependency_prefixes,
            bucket=event["bucket"],
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
            "analysis_id": event["analysis_id"],
            "files_written": result.files_written,
        }

    return handler
