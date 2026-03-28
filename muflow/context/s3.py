"""S3 workflow context."""

from __future__ import annotations

import io
from typing import IO, Any

import xarray as xr

from muflow.io.json import dumps_json, loads_json
from muflow.io.xarray import (
    load_xarray_from_bytes,
    save_xarray_to_bytes,
)


class S3WorkflowContext:
    """WorkflowContext backed by S3.

    For use in Lambda or Batch where Django is not available.
    Requires boto3 to be installed.

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
        Boto3 S3 client. If not provided, one will be created.
    allowed_outputs : set[str] | None, optional
        Set of filenames this context is allowed to write.
        None means all writes allowed (default for backward compatibility).
        Empty set means read-only (used for dependency contexts).
    """

    def __init__(
        self,
        storage_prefix: str,
        kwargs: dict,
        dependency_prefixes: dict[str, str],
        bucket: str,
        s3_client=None,
        allowed_outputs: set[str] | None = None,
    ):
        self._prefix = storage_prefix
        self._kwargs = kwargs
        self._dep_prefixes = dependency_prefixes
        self._bucket = bucket
        self._allowed_outputs = allowed_outputs

        if s3_client is None:
            try:
                import boto3
                self._s3 = boto3.client("s3")
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3WorkflowContext. "
                    "Install with: pip install muflow[s3]"
                )
        else:
            self._s3 = s3_client

    @property
    def storage_prefix(self) -> str:
        """Return the S3 key prefix."""
        return self._prefix

    @property
    def kwargs(self) -> dict:
        """Return workflow parameters."""
        return self._kwargs

    @property
    def allowed_outputs(self) -> set[str] | None:
        """Return set of allowed output filenames."""
        return self._allowed_outputs

    def _key(self, filename: str) -> str:
        """Get full S3 key for a file."""
        return f"{self._prefix}/{filename}"

    def _validate_write(self, filename: str) -> None:
        """Raise if filename is not in allowed_outputs."""
        if self._allowed_outputs is None:
            return  # No restriction
        if filename not in self._allowed_outputs:
            if not self._allowed_outputs:
                raise PermissionError(
                    f"Attempted to write '{filename}' to a read-only context"
                )
            raise PermissionError(
                f"Workflow attempted to write '{filename}' but only "
                f"{sorted(self._allowed_outputs)} are declared in Outputs"
            )

    def save_file(self, filename: str, data: bytes) -> None:
        """Save raw bytes to S3."""
        self._validate_write(filename)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(filename),
            Body=data,
        )

    def save_json(self, filename: str, data: Any) -> None:
        """Save data as JSON to S3."""
        self._validate_write(filename)
        body = dumps_json(data).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(filename),
            Body=body,
            ContentType="application/json",
        )

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        """Save an xarray Dataset as NetCDF to S3."""
        self._validate_write(filename)
        data = save_xarray_to_bytes(dataset)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(filename),
            Body=data,
            ContentType="application/x-netcdf",
        )

    def open_file(self, filename: str, mode: str = "r") -> IO:
        """Open a file from S3 for reading."""
        obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(filename))
        raw = obj["Body"].read()
        if "b" in mode:
            return io.BytesIO(raw)
        else:
            return io.StringIO(raw.decode("utf-8"))

    def read_file(self, filename: str) -> bytes:
        """Read raw bytes from S3."""
        obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(filename))
        return obj["Body"].read()

    def read_json(self, filename: str) -> Any:
        """Read and parse a JSON file from S3."""
        data = self.read_file(filename)
        return loads_json(data.decode("utf-8"))

    def read_xarray(self, filename: str) -> xr.Dataset:
        """Read a NetCDF file from S3 as xarray Dataset."""
        data = self.read_file(filename)
        return load_xarray_from_bytes(data)

    def exists(self, filename: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(filename))
            return True
        except self._s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def dependency(self, key: str) -> S3WorkflowContext:
        """Get a read-only context for accessing a dependency's outputs."""
        if key not in self._dep_prefixes:
            raise KeyError(f"Unknown dependency: {key}")
        return S3WorkflowContext(
            storage_prefix=self._dep_prefixes[key],
            kwargs={},
            dependency_prefixes={},
            bucket=self._bucket,
            s3_client=self._s3,
            allowed_outputs=set(),  # Read-only
        )

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress by writing to S3.

        Writes a _progress.json file that can be polled by the frontend
        or orchestration layer to track workflow progress.

        Parameters
        ----------
        current : int
            Current progress value.
        total : int
            Total value for progress calculation.
        message : str
            Optional progress message.
        """
        from datetime import datetime, timezone
        progress_data = {
            "current": current,
            "total": total,
            "message": message,
            "percentage": (current / total * 100) if total > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Use put_object directly to avoid allowed_outputs validation
        # since _progress.json is an internal file
        body = dumps_json(progress_data).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key("_progress.json"),
            Body=body,
            ContentType="application/json",
        )
