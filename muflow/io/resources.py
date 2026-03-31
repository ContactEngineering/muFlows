"""I/O utilities for resource fetching.

This module provides utilities for transparently fetching resources from
local files or URLs. Resources are identified by URI strings:

- `https://example.com/file.nc` - Fetch from URL
- `http://example.com/file.nc` - Fetch from URL
- `file:///path/to/file.nc` - Local file (explicit)
- `/path/to/file.nc` - Local file (implicit, backward compatible)
- `relative/path.nc` - Local file (implicit, backward compatible)

For URLs, resources are downloaded to a temporary file and the path to
that file is returned. The caller is responsible for cleanup if needed.
"""

from __future__ import annotations

import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


def is_url(uri: str) -> bool:
    """Check if a URI is a remote URL.

    Parameters
    ----------
    uri : str
        Resource URI.

    Returns
    -------
    bool
        True if the URI is an HTTP or HTTPS URL.
    """
    parsed = urlparse(uri)
    return parsed.scheme in ("http", "https")


def is_local_file(uri: str) -> bool:
    """Check if a URI refers to a local file.

    Parameters
    ----------
    uri : str
        Resource URI.

    Returns
    -------
    bool
        True if the URI is a local file path.
    """
    parsed = urlparse(uri)
    return parsed.scheme in ("", "file")


def resolve_uri(uri: str) -> str:
    """Resolve a URI to a local file path.

    For local files, returns the path directly.
    For URLs, downloads to a temporary file and returns that path.

    Parameters
    ----------
    uri : str
        Resource URI (URL or file path).

    Returns
    -------
    str
        Path to local file.

    Raises
    ------
    ValueError
        If the URI scheme is not supported.
    urllib.error.URLError
        If URL fetching fails.
    """
    parsed = urlparse(uri)

    if parsed.scheme in ("http", "https"):
        return _fetch_url(uri)
    elif parsed.scheme == "file":
        # file:///path/to/file -> /path/to/file
        return parsed.path
    elif parsed.scheme == "":
        # No scheme = local file path
        return uri
    else:
        raise ValueError(
            f"Unsupported URI scheme: {parsed.scheme}. "
            f"Use http://, https://, file://, or a local path."
        )


def _fetch_url(url: str) -> str:
    """Fetch a URL to a temporary file.

    Parameters
    ----------
    url : str
        URL to fetch.

    Returns
    -------
    str
        Path to temporary file containing the downloaded content.
    """
    _log.info(f"Fetching resource from URL: {url}")

    # Determine file extension from URL
    parsed = urlparse(url)
    path = Path(parsed.path)
    suffix = path.suffix or ""

    # Download to temporary file
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, prefix="sds_resource_"
    ) as tmp:
        with urllib.request.urlopen(url) as response:
            tmp.write(response.read())
        _log.debug(f"Downloaded {url} to {tmp.name}")
        return tmp.name


class ResourceManager:
    """Context manager for resource fetching with automatic cleanup.

    Use this when you need to fetch multiple resources and want automatic
    cleanup of any temporary files created for URL downloads.

    Example
    -------
    >>> with ResourceManager() as rm:
    ...     topo_path = rm.resolve("https://example.com/surface.nc")
    ...     model_path = rm.resolve("/local/model.nc")
    ...     # Use the paths...
    ... # Temporary files are cleaned up automatically
    """

    def __init__(self):
        self._temp_files: list[str] = []

    def __enter__(self) -> "ResourceManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def resolve(self, uri: str) -> str:
        """Resolve a URI to a local file path.

        Parameters
        ----------
        uri : str
            Resource URI.

        Returns
        -------
        str
            Path to local file.
        """
        if is_url(uri):
            path = _fetch_url(uri)
            self._temp_files.append(path)
            return path
        else:
            return resolve_uri(uri)

    def cleanup(self):
        """Remove any temporary files created during resolution."""
        import os

        for path in self._temp_files:
            try:
                os.unlink(path)
                _log.debug(f"Cleaned up temporary file: {path}")
            except OSError:
                pass
        self._temp_files.clear()
