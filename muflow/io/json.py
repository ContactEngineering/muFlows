"""JSON utilities with support for NaN, Infinity, and numpy types."""

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np


def _encode_special_float(value: float) -> str | None:
    """Return a string sentinel for NaN/Inf floats, or None."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return None


def _encode_np_floating(value: np.floating) -> Any:
    """Convert a numpy float to a Python object."""
    sentinel = _encode_special_float(float(value))
    return sentinel if sentinel is not None else float(value)


class ExtendedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NaN, Infinity, numpy types, and dates.

    - NaN becomes "NaN" (string)
    - Infinity becomes "Infinity" (string)
    - -Infinity becomes "-Infinity" (string)
    - numpy arrays become lists
    - numpy scalars become Python scalars
    - datetime/date become ISO format strings
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, float):
            sentinel = _encode_special_float(obj)
            if sentinel is not None:
                return sentinel
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return _encode_np_floating(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

    def encode(self, obj: Any) -> str:
        # Handle top-level floats
        if isinstance(obj, float):
            sentinel = _encode_special_float(obj)
            if sentinel is not None:
                return json.dumps(sentinel)
        return super().encode(obj)

    def iterencode(self, obj: Any, _one_shot: bool = False):
        # Override to handle floats in nested structures
        return super().iterencode(self._convert_floats(obj), _one_shot)

    def _convert_floats(self, obj: Any) -> Any:
        """Recursively convert special float values."""
        if isinstance(obj, dict):
            return {k: self._convert_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_floats(item) for item in obj]
        if isinstance(obj, float):
            sentinel = _encode_special_float(obj)
            return sentinel if sentinel is not None else obj
        if isinstance(obj, np.floating):
            return _encode_np_floating(obj)
        if isinstance(obj, np.ndarray):
            return self._convert_floats(obj.tolist())
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj


# Constants for decoding special float values
JSON_FLOAT_CONSTANTS = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}


def _decode_floats(obj: Any) -> Any:
    """Recursively decode special float string values back to floats."""
    if isinstance(obj, dict):
        return {k: _decode_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_floats(item) for item in obj]
    elif isinstance(obj, str) and obj in JSON_FLOAT_CONSTANTS:
        return JSON_FLOAT_CONSTANTS[obj]
    return obj


def dumps_json(obj: Any, **kwargs) -> str:
    """Serialize object to JSON string with extended type support.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    **kwargs
        Additional arguments passed to json.dumps.

    Returns
    -------
    str
        JSON string.
    """
    kwargs.setdefault("cls", ExtendedJSONEncoder)
    return json.dumps(obj, **kwargs)


def loads_json(s: str, **kwargs) -> Any:
    """Deserialize JSON string with special float value support.

    Parameters
    ----------
    s : str
        JSON string to deserialize.
    **kwargs
        Additional arguments passed to json.loads.

    Returns
    -------
    Any
        Deserialized object with "NaN", "Infinity", "-Infinity" converted
        back to float values.
    """
    obj = json.loads(s, **kwargs)
    return _decode_floats(obj)
