"""S3 storage backend."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import IO, Any

import xarray as xr

from muflow.io.json import dumps_json, loads_json
from muflow.io.xarray import load_xarray_from_bytes, save_xarray_to_bytes
from muflow.storage.base import compute_prefix, validate_filename, validate_writable


class S3StorageBackend:
    """Storage backend backed by AWS S3.

    Parameters
    ----------
    storage_prefix : str
        S3 key prefix for files.
    bucket : str
        S3 bucket name.
    s3_client : optional
        Boto3 S3 client.  If not provided, one will be created.
    hash_dict : dict, optional
        When provided, *storage_prefix* is computed from the hash dict
        instead of using the explicit *storage_prefix* argument.
    base_prefix : str
        Base prefix passed to ``compute_prefix``.  Only used when
        *hash_dict* is provided.
    """

    def __init__(
        self,
        storage_prefix: str = "",
        bucket: str = "",
        s3_client=None,
        hash_dict: dict = None,
        base_prefix: str = "muflow",
    ):
        if hash_dict is not None:
            self._prefix = compute_prefix(hash_dict, base_prefix)
        else:
            self._prefix = storage_prefix
        self._bucket = bucket
        self._written_files: set[str] = set()

        if s3_client is None:
            try:
                import boto3
                self._s3 = boto3.client("s3")
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3StorageBackend. "
                    "Install with: pip install muflow[s3]"
                )
        else:
            self._s3 = s3_client

    @property
    def storage_prefix(self) -> str:
        return self._prefix

    @property
    def written_files(self) -> frozenset[str]:
        return frozenset(self._written_files)

    def _key(self, filename: str) -> str:
        return f"{self._prefix}/{filename}"

    # ── Write methods ───────────────────────────────────────────────────

    def save_file(self, filename: str, data: bytes) -> None:
        validate_filename(filename)
        validate_writable(filename, self._written_files)
        self._s3.put_object(
            Bucket=self._bucket, Key=self._key(filename), Body=data,
        )
        self._written_files.add(filename)

    def save_json(self, filename: str, data: Any) -> None:
        validate_filename(filename)
        validate_writable(filename, self._written_files)
        body = dumps_json(data).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket, Key=self._key(filename),
            Body=body, ContentType="application/json",
        )
        self._written_files.add(filename)

    def save_xarray(self, filename: str, dataset: xr.Dataset) -> None:
        validate_filename(filename)
        validate_writable(filename, self._written_files)
        data = save_xarray_to_bytes(dataset)
        self._s3.put_object(
            Bucket=self._bucket, Key=self._key(filename),
            Body=data, ContentType="application/x-netcdf",
        )
        self._written_files.add(filename)

    # ── Read methods ────────────────────────────────────────────────────

    def open_file(self, filename: str, mode: str = "r") -> IO:
        validate_filename(filename)
        obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(filename))
        raw = obj["Body"].read()
        if "b" in mode:
            return io.BytesIO(raw)
        return io.StringIO(raw.decode("utf-8"))

    def read_file(self, filename: str) -> bytes:
        validate_filename(filename)
        obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(filename))
        return obj["Body"].read()

    def read_json(self, filename: str) -> Any:
        validate_filename(filename)
        data = self.read_file(filename)
        return loads_json(data.decode("utf-8"))

    def read_xarray(self, filename: str) -> xr.Dataset:
        validate_filename(filename)
        data = self.read_file(filename)
        return load_xarray_from_bytes(data)

    def exists(self, filename: str) -> bool:
        validate_filename(filename)
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(filename))
            return True
        except self._s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    # ── Caching ─────────────────────────────────────────────────────────

    def is_cached(self) -> bool:
        """Check if results already exist (``manifest.json`` present)."""
        return self.exists("manifest.json")

    # ── Manifest ────────────────────────────────────────────────────────

    def write_manifest(self) -> None:
        """Write ``manifest.json`` listing all files written in this session."""
        manifest = {
            "files": sorted(self._written_files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        body = dumps_json(manifest).encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket, Key=self._key("manifest.json"),
            Body=body, ContentType="application/json",
        )
