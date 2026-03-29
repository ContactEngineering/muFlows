"""Local filesystem storage backend."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Union

import xarray as xr

from muflow.io.json import dumps_json, loads_json
from muflow.io.xarray import load_xarray_from_file, save_xarray_to_file
from muflow.storage.base import compute_prefix, validate_filename, validate_writable


class LocalStorageBackend:
    """Storage backend backed by a local directory.

    Parameters
    ----------
    path : str or Path
        Root directory for file storage.  Created if it does not exist.
    hash_dict : dict, optional
        When provided together with *path*, the actual storage directory
        becomes ``path / compute_prefix(hash_dict)``.  This enables
        content-addressed storage where the directory name is derived
        from the computation identity.
    base_prefix : str
        Base prefix passed to ``compute_prefix``.  Only used when
        *hash_dict* is provided.
    """

    def __init__(
        self,
        path: Union[str, Path],
        hash_dict: dict = None,
        base_prefix: str = "muflow",
    ):
        if hash_dict is not None:
            self._path = Path(path) / compute_prefix(hash_dict, base_prefix)
        else:
            self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._written_files: set[str] = set()

    @property
    def storage_prefix(self) -> str:
        return str(self._path)

    @property
    def written_files(self) -> frozenset[str]:
        return frozenset(self._written_files)

    def _full_path(self, filename: str) -> Path:
        return self._path / filename

    # ── Write methods ───────────────────────────────────────────────────

    def save_file(self, filename: str, data: bytes) -> None:
        validate_filename(filename)
        validate_writable(filename, self._written_files)
        path = self._full_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._written_files.add(filename)

    def save_json(self, filename: str, data: Any) -> None:
        validate_filename(filename)
        validate_writable(filename, self._written_files)
        path = self._full_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_json(data, indent=2))
        self._written_files.add(filename)

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        validate_filename(filename)
        validate_writable(filename, self._written_files)
        path = self._full_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_xarray_to_file(dataset, str(path))
        self._written_files.add(filename)

    # ── Read methods ────────────────────────────────────────────────────

    def open_file(self, filename: str, mode: str = "r") -> IO:
        validate_filename(filename)
        return open(self._full_path(filename), mode)

    def read_file(self, filename: str) -> bytes:
        validate_filename(filename)
        return self._full_path(filename).read_bytes()

    def read_json(self, filename: str) -> Any:
        validate_filename(filename)
        return loads_json(self._full_path(filename).read_text())

    def read_xarray(self, filename: str) -> xr.Dataset:
        validate_filename(filename)
        return load_xarray_from_file(str(self._full_path(filename)))

    def exists(self, filename: str) -> bool:
        validate_filename(filename)
        return self._full_path(filename).exists()

    # ── Caching ─────────────────────────────────────────────────────────

    def is_cached(self) -> bool:
        """Check if results already exist (``manifest.json`` present)."""
        return self._full_path("manifest.json").exists()

    # ── Manifest ────────────────────────────────────────────────────────

    def write_manifest(self) -> None:
        """Write ``manifest.json`` listing all files written in this session."""
        manifest = {
            "files": sorted(self._written_files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = self._full_path("manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_json(manifest, indent=2))
