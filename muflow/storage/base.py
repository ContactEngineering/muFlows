"""Storage backend protocol and safety utilities.

A StorageBackend handles raw file I/O with built-in safety features:
- Path traversal protection (rejects ``..`` and absolute paths)
- Write-once semantics (no overwrites)
- Protected files (``context.json`` and ``manifest.json`` cannot be written
  by workflows)
- Manifest generation (list of all files written during a session)

Implementations include ``LocalStorageBackend`` (filesystem) and
``S3StorageBackend`` (AWS S3).  Domain-specific contexts (e.g.
``TopographyContext`` in sds-workflows) wrap a storage backend and add
subject-loading behaviour.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import IO, Any, Protocol, runtime_checkable

import xarray as xr


# ── Reserved / protected filenames ──────────────────────────────────────────

PROTECTED_FILES = frozenset({"context.json", "manifest.json"})


# ── Filename validation ─────────────────────────────────────────────────────

def validate_filename(filename: str) -> None:
    """Raise ``ValueError`` if *filename* is unsafe.

    A filename is considered unsafe if it:
    - is empty,
    - is an absolute path,
    - contains ``..`` as a path component, or
    - resolves to a path outside the storage root after normalisation.

    Parameters
    ----------
    filename : str
        The filename to validate.

    Raises
    ------
    ValueError
        If the filename is unsafe.
    """
    if not filename:
        raise ValueError("Filename must not be empty.")
    if os.path.isabs(filename):
        raise ValueError(
            f"Absolute paths are not allowed: '{filename}'"
        )
    # Normalise and check for path traversal
    normalised = os.path.normpath(filename)
    if normalised.startswith("..") or normalised.startswith(os.sep):
        raise ValueError(
            f"Path traversal is not allowed: '{filename}'"
        )


def validate_writable(filename: str, written_files: set[str]) -> None:
    """Raise if *filename* is protected or has already been written.

    Parameters
    ----------
    filename : str
        The filename to validate.
    written_files : set[str]
        Set of filenames that have already been written in this session.

    Raises
    ------
    PermissionError
        If the filename is protected (``context.json`` or ``manifest.json``).
    FileExistsError
        If the file has already been written in this session.
    """
    if filename in PROTECTED_FILES:
        raise PermissionError(
            f"'{filename}' is a protected file and cannot be written by "
            f"workflows."
        )
    if filename in written_files:
        raise FileExistsError(
            f"'{filename}' has already been written.  Files can only be "
            f"written once."
        )


# ── Content-addressed prefix computation ────────────────────────────────────

def compute_prefix(hash_dict: dict, base_prefix: str = "muflow") -> str:
    """Compute a content-addressed storage prefix from a dictionary.

    The dictionary is serialised as sorted JSON and hashed with SHA-256.
    The ``workflow`` key (if present) is used as a human-readable path
    component.

    Parameters
    ----------
    hash_dict : dict
        Dictionary of fields that uniquely identify a computation.
    base_prefix : str
        Base path prefix.  Default is ``"muflow"``.

    Returns
    -------
    str
        Storage prefix like ``"muflow/my.workflow/a1b2c3d4e5f6g7h8"``.
    """
    content = json.dumps(hash_dict, sort_keys=True)
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    label = hash_dict.get("workflow", "unknown")
    return f"{base_prefix}/{label}/{h}"


# ── Protocol ────────────────────────────────────────────────────────────────

@runtime_checkable
class StorageBackend(Protocol):
    """Abstract interface for workflow file storage.

    Implementations must enforce the safety rules (path traversal, write-once,
    protected files) using the ``validate_filename`` and ``validate_writable``
    helpers defined in this module.
    """

    @property
    def storage_prefix(self) -> str:
        """Root path or S3 prefix for this storage."""
        ...

    @property
    def written_files(self) -> frozenset[str]:
        """Set of filenames written during this session."""
        ...

    # ── Write methods ───────────────────────────────────────────────────

    def save_file(self, filename: str, data: bytes) -> None:
        """Save raw bytes."""
        ...

    def save_json(self, filename: str, data: Any) -> None:
        """Save data as JSON."""
        ...

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        """Save an xarray Dataset as NetCDF."""
        ...

    # ── Read methods ────────────────────────────────────────────────────

    def open_file(self, filename: str, mode: str = "r") -> IO:
        """Open a file for reading."""
        ...

    def read_file(self, filename: str) -> bytes:
        """Read raw bytes."""
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

    # ── Caching ──────────────────────────────────────────────────────────

    def is_cached(self) -> bool:
        """Check if results already exist (``manifest.json`` present)."""
        ...

    # ── Manifest ────────────────────────────────────────────────────────

    def write_manifest(self) -> None:
        """Write ``manifest.json`` listing all files written in this session.

        Called by the executor after the workflow function returns (or raises).
        """
        ...
