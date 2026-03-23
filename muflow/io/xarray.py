"""xarray I/O utilities for muflow."""

import io
import tempfile
import os
from typing import Union

import xarray as xr


def save_xarray_to_bytes(
    dataset: xr.Dataset,
    format: str = "NETCDF3_CLASSIC",
    engine: str = "scipy",
) -> bytes:
    """Serialize an xarray Dataset to bytes.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to serialize.
    format : str, optional
        NetCDF format. Default is "NETCDF3_CLASSIC" for maximum compatibility.
    engine : str, optional
        Engine to use for writing. Default is "scipy".

    Returns
    -------
    bytes
        Serialized dataset as bytes.

    Notes
    -----
    Uses a temporary file because NetCDF libraries typically require
    seekable file handles that they close after writing.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as f:
        temp_path = f.name

    try:
        dataset.to_netcdf(temp_path, format=format, engine=engine)
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_xarray_from_bytes(
    data: bytes,
    engine: str = "scipy",
) -> xr.Dataset:
    """Load an xarray Dataset from bytes.

    Parameters
    ----------
    data : bytes
        Serialized NetCDF data.
    engine : str, optional
        Engine to use for reading. Default is "scipy".

    Returns
    -------
    xr.Dataset
        The loaded dataset.

    Notes
    -----
    Uses xr.load_dataset (not open_dataset) to fully load data into memory,
    avoiding issues with file handle lifetime.
    """
    return xr.load_dataset(io.BytesIO(data), engine=engine)


def save_xarray_to_file(
    dataset: xr.Dataset,
    path: str,
    format: str = "NETCDF3_CLASSIC",
    engine: str = "scipy",
) -> None:
    """Save an xarray Dataset to a file path.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to save.
    path : str
        Path to write to.
    format : str, optional
        NetCDF format. Default is "NETCDF3_CLASSIC".
    engine : str, optional
        Engine to use for writing. Default is "scipy".
    """
    dataset.to_netcdf(path, format=format, engine=engine)


def load_xarray_from_file(
    path: str,
    engine: str = "scipy",
) -> xr.Dataset:
    """Load an xarray Dataset from a file path.

    Parameters
    ----------
    path : str
        Path to read from.
    engine : str, optional
        Engine to use for reading. Default is "scipy".

    Returns
    -------
    xr.Dataset
        The loaded dataset.
    """
    return xr.load_dataset(path, engine=engine)
