"""Unified workflow context.

A single WorkflowContext class that works with any StorageBackend.
"""

from __future__ import annotations

from typing import IO, Any, Callable, Dict, Optional

import xarray as xr

from muflow.storage.base import StorageBackend


def _print_progress(current: int, total: int, message: str) -> None:
    """Default progress reporter that prints to stdout."""
    pct = current / total * 100 if total > 0 else 0
    # Right-align percentage in 6 characters (e.g., "100.0%")
    pct_str = f"{pct:.1f}%".rjust(6)
    print(f"  [{pct_str}] {message}")


class WorkflowContext:
    """Unified workflow context that works with any StorageBackend.

    This context class handles file I/O (delegated to a StorageBackend),
    dependency access, progress reporting, and workflow parameters.

    Parameters
    ----------
    storage : StorageBackend
        The storage backend for file I/O (LocalStorageBackend, S3StorageBackend, etc.)
    kwargs : dict
        Raw workflow parameters.
    dependency_storages : dict[str, StorageBackend], optional
        Mapping from dependency key to storage backend for that dependency.
    progress_reporter : callable, optional
        Function called with (current, total, message) for progress updates.
        Defaults to printing to stdout.

    Attributes
    ----------
    kwargs : Any
        Validated parameters (pydantic model), set by the executor after
        validation. ``None`` if no parameter model is registered.

    Example
    -------
    >>> from muflow.storage import LocalStorageBackend
    >>> from muflow.context import WorkflowContext
    >>>
    >>> storage = LocalStorageBackend("/tmp/output")
    >>> dep_storages = {"dep1": LocalStorageBackend("/tmp/dep1")}
    >>> ctx = WorkflowContext(storage, {"param": "value"}, dep_storages)
    >>>
    >>> ctx.save_json("result.json", {"status": "ok"})
    >>> dep_ctx = ctx.dependency("dep1")
    >>> data = dep_ctx.read_json("features.json")
    """

    def __init__(
        self,
        storage: StorageBackend,
        kwargs: dict,
        dependency_storages: Optional[Dict[str, StorageBackend]] = None,
        progress_reporter: Optional[Callable[[int, int, str], None]] = None,
    ):
        self._storage = storage
        self._raw_kwargs = kwargs
        self._dependency_storages = dependency_storages or {}
        self._progress_reporter = progress_reporter or _print_progress
        self._kwargs = kwargs  # Set by executor for function-based workflows

    # ── Parameters ──────────────────────────────────────────────────────

    @property
    def kwargs(self) -> Any:
        """Validated parameters (pydantic model), or ``None``."""
        return self._kwargs

    # ── Storage ─────────────────────────────────────────────────────────

    @property
    def storage(self) -> StorageBackend:
        """Return the underlying storage backend."""
        return self._storage

    @property
    def storage_prefix(self) -> str:
        """Return the storage prefix (path or S3 key prefix)."""
        return self._storage.storage_prefix

    # ── File I/O (delegated to storage backend) ─────────────────────────

    def save_file(self, filename: str, data: bytes) -> None:
        """Save raw bytes to a file."""
        self._storage.save_file(filename, data)

    def save_json(
        self, filename: str, data: Any, allow_protected: bool = False
    ) -> None:
        """Save data as JSON."""
        self._storage.save_json(filename, data, allow_protected=allow_protected)

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        """Save an xarray Dataset as NetCDF."""
        self._storage.save_xarray(filename, dataset)

    def open_file(self, filename: str, mode: str = "r") -> IO:
        """Open a file for reading."""
        return self._storage.open_file(filename, mode)

    def read_file(self, filename: str) -> bytes:
        """Read raw bytes from a file."""
        return self._storage.read_file(filename)

    def read_json(self, filename: str) -> Any:
        """Read and parse a JSON file."""
        return self._storage.read_json(filename)

    def read_xarray(self, filename: str) -> xr.Dataset:
        """Read a NetCDF file as xarray Dataset."""
        return self._storage.read_xarray(filename)

    def exists(self, filename: str) -> bool:
        """Check if a file exists."""
        return self._storage.exists(filename)

    # ── Dependency access ───────────────────────────────────────────────

    def has_dependency(self, key: str) -> bool:
        """Check if a dependency is available.

        Parameters
        ----------
        key : str
            The dependency key (e.g., "loo_0", "surface_0").

        Returns
        -------
        bool
            True if the dependency is available, False otherwise.
        """
        return key in self._dependency_storages

    def dependency(self, key: str) -> "WorkflowContext":
        """Get a read-only context for accessing a dependency's outputs.

        Parameters
        ----------
        key : str
            The dependency key (e.g., "surface_0", "feature_matrix").

        Returns
        -------
        WorkflowContext
            A context for reading the dependency's output files.

        Raises
        ------
        KeyError
            If the dependency key is not found.
        """
        if key not in self._dependency_storages:
            raise KeyError(f"Unknown dependency: {key}")
        return WorkflowContext(
            storage=self._dependency_storages[key],
            kwargs={},
            dependency_storages={},
            progress_reporter=self._progress_reporter,
        )

    # ── Progress reporting ──────────────────────────────────────────────

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress.

        Parameters
        ----------
        current : int
            Current step number.
        total : int
            Total number of steps.
        message : str, optional
            Progress message.
        """
        self._progress_reporter(current, total, message)


def create_local_context(
    path: str,
    kwargs: dict,
    dependency_paths: dict = None,
    progress_reporter: Optional[Callable[[int, int, str], None]] = None,
) -> WorkflowContext:
    """Create a WorkflowContext backed by local filesystem.

    Parameters
    ----------
    path : str
        Local directory path for storing files.
    kwargs : dict
        Workflow parameters.
    dependency_paths : dict[str, str], optional
        Mapping from dependency key to local path.
    progress_reporter : callable, optional
        Function called with (current, total, message) for progress updates.
        Defaults to printing to stdout with right-aligned percentage.

    Returns
    -------
    WorkflowContext
        A context backed by LocalStorageBackend.

    Example
    -------
    >>> ctx = create_local_context(
    ...     path="/tmp/output",
    ...     kwargs={"param": "value"},
    ...     dependency_paths={"dep1": "/tmp/dep1"},
    ... )
    """
    from muflow.storage import LocalStorageBackend

    storage = LocalStorageBackend(path)
    dep_storages = {}
    if dependency_paths:
        dep_storages = {
            key: LocalStorageBackend(p) for key, p in dependency_paths.items()
        }
    return WorkflowContext(
        storage=storage,
        kwargs=kwargs,
        dependency_storages=dep_storages,
        progress_reporter=progress_reporter,
    )
