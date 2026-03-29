"""S3 workflow context."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import IO, Any

import xarray as xr

from muflow.context.parameterized import ParameterizedMixin
from muflow.io.json import dumps_json
from muflow.storage import S3StorageBackend


class S3WorkflowContext(ParameterizedMixin):
    """WorkflowContext backed by S3.

    Delegates all file I/O to an ``S3StorageBackend``.  Inherits
    ``kwargs`` and ``parameters`` from ``ParameterizedMixin``.

    Parameters
    ----------
    storage_prefix : str
        S3 key prefix for this workflow's output files.
    kwargs : dict
        Workflow parameters.
    dependency_prefixes : dict[str, str]
        Mapping from dependency key to S3 prefix.
    bucket : str
        S3 bucket name.
    s3_client : optional
        Boto3 S3 client.  If not provided, one will be created.
    storage : S3StorageBackend, optional
        Pre-created storage backend.
    """

    def __init__(
        self,
        storage_prefix: str,
        kwargs: dict,
        dependency_prefixes: dict[str, str],
        bucket: str,
        s3_client=None,
        storage: S3StorageBackend = None,
    ):
        self._storage = storage or S3StorageBackend(
            storage_prefix, bucket, s3_client
        )
        self._kwargs = kwargs
        self._dep_prefixes = dependency_prefixes
        self._bucket = bucket
        self._s3 = self._storage._s3
        self._parameters = None

    @property
    def storage(self) -> S3StorageBackend:
        return self._storage

    @property
    def storage_prefix(self) -> str:
        return self._storage.storage_prefix

    # ── File I/O (delegated to storage backend) ─────────────────────────

    def save_file(self, filename: str, data: bytes) -> None:
        self._storage.save_file(filename, data)

    def save_json(self, filename: str, data: Any) -> None:
        self._storage.save_json(filename, data)

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        self._storage.save_xarray(filename, dataset)

    def open_file(self, filename: str, mode: str = "r") -> IO:
        return self._storage.open_file(filename, mode)

    def read_file(self, filename: str) -> bytes:
        return self._storage.read_file(filename)

    def read_json(self, filename: str) -> Any:
        return self._storage.read_json(filename)

    def read_xarray(self, filename: str) -> xr.Dataset:
        return self._storage.read_xarray(filename)

    def exists(self, filename: str) -> bool:
        return self._storage.exists(filename)

    # ── Dependency access ───────────────────────────────────────────────

    def dependency(self, key: str) -> S3WorkflowContext:
        if key not in self._dep_prefixes:
            raise KeyError(f"Unknown dependency: {key}")
        return S3WorkflowContext(
            storage_prefix=self._dep_prefixes[key],
            kwargs={},
            dependency_prefixes={},
            bucket=self._bucket,
            s3_client=self._s3,
        )

    # ── Progress reporting ──────────────────────────────────────────────

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        progress_data = {
            "current": current,
            "total": total,
            "message": message,
            "percentage": (current / total * 100) if total > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        body = dumps_json(progress_data).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=f"{self.storage_prefix}/_progress.json",
            Body=body,
            ContentType="application/json",
        )
