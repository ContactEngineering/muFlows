"""Storage backend abstractions.

This package provides the ``StorageBackend`` protocol and its implementations.
Storage backends handle raw file I/O with safety features (path traversal
protection, write-once semantics, protected files, manifest generation).

Modules
-------
base
    ``StorageBackend`` protocol, ``validate_filename``, ``validate_writable``.
local
    ``LocalStorageBackend`` — backed by the local filesystem.
s3
    ``S3StorageBackend`` — backed by AWS S3 via boto3.
"""

from muflow.storage.base import (
    PROTECTED_FILES,
    StorageBackend,
    compute_prefix,
    validate_filename,
    validate_writable,
)
from muflow.storage.local import LocalStorageBackend
from muflow.storage.s3 import S3StorageBackend

__all__ = [
    "PROTECTED_FILES",
    "StorageBackend",
    "LocalStorageBackend",
    "S3StorageBackend",
    "compute_prefix",
    "validate_filename",
    "validate_writable",
]
