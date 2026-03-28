"""Local filesystem workflow context."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Union

import xarray as xr

from muflow.io.json import dumps_json, loads_json
from muflow.io.xarray import (
    load_xarray_from_file,
    save_xarray_to_file,
)


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
