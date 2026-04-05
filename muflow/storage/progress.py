"""Progress checking across storage backends.

A node is complete when ``manifest.json`` exists at its storage prefix.
:class:`ProgressChecker` implementations check multiple prefixes in one
call, returning the subset that are complete.

Unlike :class:`~muflow.storage.base.StorageBackend` (which is bound to a
single prefix), a ``ProgressChecker`` is bound to a *storage configuration*
(e.g. an S3 bucket) and operates on any number of prefixes.  This makes it
suitable for serializing inside :class:`~muflow.backends.handle.PlanHandle`.

Implementations
---------------
:class:`LocalProgressChecker`
    Checks ``{prefix}/manifest.json`` via ``os.path.exists``.
:class:`S3ProgressChecker`
    Checks ``{prefix}/manifest.json`` via ``s3.head_object`` (one HEAD
    request per prefix; sequential within AWS is typically 10–50 ms each).

Adding a new storage backend requires only a new class implementing the
protocol and a branch in :func:`make_progress_checker`.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressChecker(Protocol):
    """Protocol for checking node completion across multiple prefixes.

    Each prefix corresponds to one task node.  A node is considered
    complete when ``manifest.json`` is present at that prefix.
    """

    def completed_prefixes(self, prefixes: list[str]) -> set[str]:
        """Return the subset of *prefixes* that have a ``manifest.json``.

        Parameters
        ----------
        prefixes : list[str]
            Storage prefixes to check (local paths or S3 key prefixes).

        Returns
        -------
        set[str]
            The elements of *prefixes* whose ``manifest.json`` exists.
        """
        ...

    def to_config(self) -> dict:
        """Serialize checker configuration to a plain dict.

        The result is stored inside :class:`~muflow.backends.handle.PlanHandle`
        and passed back to :meth:`from_config` to reconstruct the checker.
        """
        ...

    @classmethod
    def from_config(cls, config: dict) -> "ProgressChecker":
        """Reconstruct a checker from a config dict produced by :meth:`to_config`."""
        ...


class LocalProgressChecker:
    """Progress checker for :class:`~muflow.storage.local.LocalStorageBackend`.

    Checks ``{prefix}/manifest.json`` on the local filesystem.
    No configuration is needed beyond the prefix itself.
    """

    def completed_prefixes(self, prefixes: list[str]) -> set[str]:
        return {
            p for p in prefixes
            if os.path.exists(os.path.join(p, "manifest.json"))
        }

    def to_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict) -> "LocalProgressChecker":
        return cls()


class S3ProgressChecker:
    """Progress checker for :class:`~muflow.storage.s3.S3StorageBackend`.

    Issues one ``HEAD`` request per prefix to check for ``manifest.json``.
    Within the same AWS region this is typically 10–50 ms per request;
    for plans with many nodes consider caching the result or using a
    :class:`ThreadPoolExecutor` inside :meth:`completed_prefixes`.

    Parameters
    ----------
    bucket : str
        S3 bucket name (same bucket used by the storage backend).
    """

    def __init__(self, bucket: str):
        self._bucket = bucket

    def completed_prefixes(self, prefixes: list[str]) -> set[str]:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 is required for S3ProgressChecker. "
                "Install with: pip install muflow[s3]"
            )

        s3 = boto3.client("s3")
        done: set[str] = set()
        for prefix in prefixes:
            key = f"{prefix}/manifest.json"
            try:
                s3.head_object(Bucket=self._bucket, Key=key)
                done.add(prefix)
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in ("404", "NoSuchKey"):
                    continue
                raise
        return done

    def to_config(self) -> dict:
        return {"bucket": self._bucket}

    @classmethod
    def from_config(cls, config: dict) -> "S3ProgressChecker":
        return cls(bucket=config["bucket"])


def make_progress_checker(storage_type: str, storage_config: dict) -> ProgressChecker:
    """Factory: reconstruct a :class:`ProgressChecker` from its serialized form.

    Parameters
    ----------
    storage_type : str
        ``"local"`` or ``"s3"``.
    storage_config : dict
        Config dict produced by :meth:`ProgressChecker.to_config`.

    Returns
    -------
    ProgressChecker
        A ready-to-use checker instance.

    Raises
    ------
    ValueError
        If *storage_type* is not recognised.
    """
    if storage_type == "local":
        return LocalProgressChecker.from_config(storage_config)
    if storage_type == "s3":
        return S3ProgressChecker.from_config(storage_config)
    raise ValueError(
        f"Unknown storage_type: {storage_type!r}. "
        "Expected 'local' or 's3'."
    )
