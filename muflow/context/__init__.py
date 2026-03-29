"""Workflow context abstractions.

This package provides the ``WorkflowContext`` protocol and its
implementations.  Each context wraps a ``StorageBackend`` (from
``muflow.storage``) and adds workflow-level concerns: dependency access
and progress reporting.

The base ``WorkflowContext`` protocol is agnostic of workflow parameters.
Contexts that carry parameters use ``ParameterizedMixin``.

The protocol is also domain-agnostic.  Domain-specific contexts (e.g.
``TopographyContext``, ``SurfaceContext``) live downstream in sds-workflows.

Modules
-------
base
    ``WorkflowContext`` protocol.
parameterized
    ``ParameterizedMixin`` — adds ``kwargs`` and ``parameters`` support.
local
    ``LocalFolderContext`` — backed by a ``LocalStorageBackend``.
s3
    ``S3WorkflowContext`` — backed by an ``S3StorageBackend``.
"""

from muflow.context.base import WorkflowContext
from muflow.context.local import LocalFolderContext
from muflow.context.parameterized import ParameterizedMixin
from muflow.context.s3 import S3WorkflowContext

__all__ = [
    "WorkflowContext",
    "ParameterizedMixin",
    "LocalFolderContext",
    "S3WorkflowContext",
]
