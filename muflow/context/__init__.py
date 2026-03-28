"""Workflow context abstractions.

This package provides the WorkflowContext protocol and its implementations.
The protocol is separated from the implementations so that domain-specific
contexts (e.g. TopographyContext, DjangoWorkflowContext) can extend
LocalFolderContext with additional properties without pulling in unrelated
backend dependencies.

Modules
-------
base
    WorkflowContext protocol — the abstract interface that all contexts
    implement.  Workflows depend only on this protocol.
local
    LocalFolderContext — backed by the local filesystem.  Useful for testing
    and for the ``sds-workflows`` command-line runner.
s3
    S3WorkflowContext — backed by AWS S3 via boto3.  Used by Lambda and
    Batch backends.
"""

from muflow.context.base import WorkflowContext
from muflow.context.local import LocalFolderContext
from muflow.context.s3 import S3WorkflowContext

__all__ = [
    "WorkflowContext",
    "LocalFolderContext",
    "S3WorkflowContext",
]
