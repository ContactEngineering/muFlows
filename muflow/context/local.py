"""Local filesystem workflow context."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Union

import xarray as xr

from muflow.context.parameterized import ParameterizedMixin
from muflow.storage import LocalStorageBackend


class LocalFolderContext(ParameterizedMixin):
    """WorkflowContext backed by local filesystem.

    Delegates all file I/O to a ``LocalStorageBackend``.  Inherits
    ``kwargs`` and ``parameters`` from ``ParameterizedMixin``.

    Parameters
    ----------
    path : str or Path
        Local directory path for storing files.
    kwargs : dict
        Workflow parameters.
    dependency_paths : dict[str, str], optional
        Mapping from dependency key to local path.
    storage : LocalStorageBackend, optional
        Pre-created storage backend.  If not provided, one is created from
        *path*.
    """

    def __init__(
        self,
        path: Union[str, Path],
        kwargs: dict,
        dependency_paths: dict[str, str] = None,
        storage: LocalStorageBackend = None,
    ):
        self._storage = storage or LocalStorageBackend(path)
        self._kwargs = kwargs
        self._dependency_paths = dependency_paths or {}
        self._parameters = None  # Set by executor for function-based workflows

    @property
    def storage(self) -> LocalStorageBackend:
        """Return the underlying storage backend."""
        return self._storage

    @property
    def storage_prefix(self) -> str:
        """Return the local path as a string."""
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

    def dependency(self, key: str) -> LocalFolderContext:
        """Get a read-only context for accessing a dependency's outputs."""
        if key not in self._dependency_paths:
            raise KeyError(f"Unknown dependency: {key}")
        return LocalFolderContext(
            path=self._dependency_paths[key],
            kwargs={},
            dependency_paths={},
        )

    # ── Progress reporting ──────────────────────────────────────────────

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress (prints to stdout for local testing)."""
        pct = current / total * 100 if total > 0 else 0
        print(f"[{pct:.1f}%] {message}")
