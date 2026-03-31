"""I/O utilities for muflow."""

from muflow.io.json import ExtendedJSONEncoder, dumps_json, loads_json
from muflow.io.resources import ResourceManager, is_local_file, is_url, resolve_uri
from muflow.io.xarray import load_xarray_from_bytes, save_xarray_to_bytes

__all__ = [
    "ExtendedJSONEncoder",
    "dumps_json",
    "loads_json",
    "load_xarray_from_bytes",
    "save_xarray_to_bytes",
    # Resource utilities
    "is_url",
    "is_local_file",
    "resolve_uri",
    "ResourceManager",
]
