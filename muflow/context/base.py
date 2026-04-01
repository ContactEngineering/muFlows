"""Workflow context protocol.

The WorkflowContext protocol defines the base interface that workflow
functions receive.  It provides file I/O (delegated to a storage backend),
dependency access, and progress reporting.

The protocol is deliberately domain-agnostic.  Domain-specific contexts
(e.g. TopographyContext, SurfaceContext) are defined downstream in
sds-workflows, not here.

The concrete ``WorkflowContext`` class in ``workflow.py`` implements this
protocol and adds parameter support (``kwargs`` and ``parameters``).
"""

from __future__ import annotations

from typing import IO, Any, Protocol, runtime_checkable

import xarray as xr

from muflow.storage.base import StorageBackend


@runtime_checkable
class WorkflowContext(Protocol):
    """Abstract interface for workflow execution contexts.

    A context wraps a ``StorageBackend`` and adds workflow-level concerns:
    dependency access and progress reporting.  Implementations delegate
    file I/O to the storage backend, which enforces path traversal
    protection, write-once semantics, and protected files.

    The concrete ``WorkflowContext`` class adds validated parameters via the
    ``kwargs`` property.
    """

    @property
    def kwargs(self) -> Any:
        """Validated workflow parameters (pydantic model or ``None``)."""
        ...

    @property
    def storage(self) -> StorageBackend:
        """The underlying storage backend."""
        ...

    @property
    def storage_prefix(self) -> str:
        """Root path or S3 prefix for this workflow's output files."""
        ...

    # ── File I/O (delegated to storage backend) ─────────────────────────

    def save_file(self, filename: str, data: bytes) -> None:
        """Save raw bytes to a file."""
        ...

    def save_json(
        self, filename: str, data: Any, allow_protected: bool = False
    ) -> None:
        """Save data as JSON.

        Parameters
        ----------
        filename : str
            The filename to save to.
        data : Any
            The data to serialise.
        allow_protected : bool, optional
            If True, allow writing protected files like ``context.json``.
            Default is False.
        """
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

    # ── Dependency access ───────────────────────────────────────────────
    def has_dependency(self, key: str) -> bool:
        """Check if a dependency is available."""
        ...

    def dependency(self, key: str) -> WorkflowContext:
        """Get a context for accessing a completed dependency's outputs."""
        ...

    # ── Progress reporting ──────────────────────────────────────────────

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress (may be no-op on serverless backends)."""
        ...
