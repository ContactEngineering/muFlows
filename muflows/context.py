"""Workflow context abstraction.

The WorkflowContext protocol defines the interface that workflow implementations
use for file I/O, parameter access, and dependency access. This abstraction
allows the same workflow code to run on different backends:

- DjangoWorkflowContext: Uses Django ORM (in topobank, not here)
- S3WorkflowContext: Direct S3 access via boto3 (for Lambda/Batch)
- LocalFolderContext: Local filesystem (for testing)
"""

from __future__ import annotations

import io
import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import IO, Any, Protocol, Union, runtime_checkable

import xarray as xr

from muflows.io.json import dumps_json, loads_json
from muflows.io.xarray import (
    load_xarray_from_bytes,
    load_xarray_from_file,
    save_xarray_to_bytes,
    save_xarray_to_file,
)


@runtime_checkable
class WorkflowContext(Protocol):
    """Abstract interface for workflow file I/O.

    Workflow implementations receive a WorkflowContext and use it for all
    file operations. The context handles storage backend details, allowing
    the same workflow code to run on Celery (with Django), Lambda (with S3),
    or locally (for testing).

    This is a Protocol (structural typing), so implementations don't need
    to inherit from it - they just need to implement the methods.

    Output Guards
    -------------
    The `allowed_outputs` property controls which files a workflow can write:
    - None: No restriction (backward compatibility mode)
    - set(): Read-only context (used for dependency access)
    - set(["file1.json", "file2.nc"]): Only these files can be written
    """

    @property
    def storage_prefix(self) -> str:
        """S3 key prefix or local path for this workflow's output files."""
        ...

    @property
    def kwargs(self) -> dict:
        """Parameters passed to this workflow."""
        ...

    @property
    def allowed_outputs(self) -> set[str] | None:
        """Set of filenames this workflow is allowed to write.

        Returns None if all writes are allowed (backward compatibility).
        Returns empty set if context is read-only (dependency contexts).
        Returns set of filenames if writes are restricted to declared outputs.
        """
        ...

    # File I/O
    def save_file(self, filename: str, data: bytes) -> None:
        """Save raw bytes to a file."""
        ...

    def save_json(self, filename: str, data: Any) -> None:
        """Save data as JSON."""
        ...

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        """Save an xarray Dataset as NetCDF."""
        ...

    def open_file(self, filename: str, mode: str = "r") -> IO:
        """Open a file for reading."""
        ...

    def read_file(self, filename: str) -> bytes:
        """Read raw bytes from a file."""
        ...

    def read_json(self, filename: str) -> Any:
        """Read and parse a JSON file."""
        ...

    def read_xarray(self, filename: str) -> xr.Dataset:
        """Read a NetCDF file as xarray Dataset."""
        ...

    def exists(self, filename: str) -> bool:
        """Check if a file exists."""
        ...

    # Dependency access
    def dependency(self, key: str) -> WorkflowContext:
        """Get a context for accessing a completed dependency's outputs."""
        ...

    # Progress reporting
    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress (may be no-op on serverless backends)."""
        ...


class LocalFolderContext:
    """WorkflowContext backed by local filesystem.

    Useful for testing workflows without S3 or Django.

    Parameters
    ----------
    path : str or Path
        Local directory path for storing files.
    kwargs : dict
        Workflow parameters.
    dependency_paths : dict[str, str], optional
        Mapping from dependency key to local path.
    allowed_outputs : set[str] | None, optional
        Set of filenames this context is allowed to write.
        None means all writes allowed (default for backward compatibility).
        Empty set means read-only (used for dependency contexts).
    """

    def __init__(
        self,
        path: Union[str, Path],
        kwargs: dict,
        dependency_paths: dict[str, str] = None,
        allowed_outputs: set[str] | None = None,
    ):
        self._path = Path(path)
        self._kwargs = kwargs
        self._dependency_paths = dependency_paths or {}
        self._allowed_outputs = allowed_outputs

        # Create directory if it doesn't exist
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def storage_prefix(self) -> str:
        """Return the local path as a string."""
        return str(self._path)

    @property
    def kwargs(self) -> dict:
        """Return workflow parameters."""
        return self._kwargs

    @property
    def allowed_outputs(self) -> set[str] | None:
        """Return set of allowed output filenames."""
        return self._allowed_outputs

    def _full_path(self, filename: str) -> Path:
        """Get full path to a file."""
        return self._path / filename

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
        """Save raw bytes to a file."""
        self._validate_write(filename)
        path = self._full_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save_json(self, filename: str, data: Any) -> None:
        """Save data as JSON."""
        self._validate_write(filename)
        path = self._full_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_json(data, indent=2))

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        """Save an xarray Dataset as NetCDF."""
        self._validate_write(filename)
        path = self._full_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_xarray_to_file(dataset, str(path))

    def open_file(self, filename: str, mode: str = "r") -> IO:
        """Open a file for reading."""
        path = self._full_path(filename)
        return open(path, mode)

    def read_file(self, filename: str) -> bytes:
        """Read raw bytes from a file."""
        return self._full_path(filename).read_bytes()

    def read_json(self, filename: str) -> Any:
        """Read and parse a JSON file."""
        text = self._full_path(filename).read_text()
        return loads_json(text)

    def read_xarray(self, filename: str) -> xr.Dataset:
        """Read a NetCDF file as xarray Dataset."""
        return load_xarray_from_file(str(self._full_path(filename)))

    def exists(self, filename: str) -> bool:
        """Check if a file exists."""
        return self._full_path(filename).exists()

    def dependency(self, key: str) -> LocalFolderContext:
        """Get a read-only context for accessing a dependency's outputs."""
        if key not in self._dependency_paths:
            raise KeyError(f"Unknown dependency: {key}")
        return LocalFolderContext(
            path=self._dependency_paths[key],
            kwargs={},
            dependency_paths={},
            allowed_outputs=set(),  # Read-only
        )

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress (prints to stdout for local testing)."""
        pct = current / total * 100 if total > 0 else 0
        print(f"[{pct:.1f}%] {message}")


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
                    "Install with: pip install muflows[s3]"
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
        """Report progress (no-op for S3/Lambda - no progress channel)."""
        pass
