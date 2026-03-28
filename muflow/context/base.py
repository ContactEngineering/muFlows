"""Workflow context protocol.

The WorkflowContext protocol defines the interface that workflow implementations
use for file I/O, parameter access, and dependency access. This abstraction
allows the same workflow code to run on different backends:

- DjangoWorkflowContext: Uses Django ORM (in topobank, not here)
- S3WorkflowContext: Direct S3 access via boto3 (for Lambda/Batch)
- LocalFolderContext: Local filesystem (for testing)
"""

from __future__ import annotations

from typing import IO, Any, Protocol, runtime_checkable

import xarray as xr


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
