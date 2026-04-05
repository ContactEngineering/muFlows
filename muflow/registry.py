"""Task registry.

Tasks are registered as plain functions using ``@register_task``.
DAG topology is declared separately via :class:`~muflow.pipeline.Pipeline`.

Example
-------
>>> from muflow.registry import register_task
>>> import pydantic
>>>
>>> class MyParams(pydantic.BaseModel):
...     threshold: float = 0.5
>>>
>>> @register_task(
...     name="myapp.my_task",
...     display_name="My Task",
...     parameters=MyParams,
... )
... def my_task(context):
...     return {"result": "done"}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

import pydantic

# ── TaskEntry ──────────────────────────────────────────────────────────


@dataclass
class IdentityKey:
    """Marker for Pydantic model fields that define the task's identity."""
    pass


@dataclass
class TaskEntry:
    """Unified descriptor for a registered task.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. ``"myapp.compute_features"``).
    fn : Callable
        The task function.  Signature: ``fn(context) -> dict | None``.
    display_name : str
        Human-readable name shown in UIs.
    queue : str
        Queue name for backend routing.
    parameters : type[pydantic.BaseModel] | None
        Pydantic model for parameter validation.  ``None`` means no
        parameters.
    outputs : type | None
        An inner ``Outputs`` class with a ``files`` dict mapping filenames
        to ``OutputFile`` descriptors.  Used for output validation.
    identity_keys : list[str] | None
        List of keys in kwargs that define the task's identity for hashing.
        If None, all kwargs are used.
    """

    name: str
    fn: Callable
    display_name: str = ""
    queue: str = "default"
    parameters: Optional[Type[pydantic.BaseModel]] = None
    outputs: Optional[Type] = None
    identity_keys: Optional[List[str]] = None


# ── Registry storage ───────────────────────────────────────────────────────

_entries_by_name: Dict[str, TaskEntry] = {}
_entries_by_display_name: Dict[str, TaskEntry] = {}


# ── Exceptions ─────────────────────────────────────────────────────────────


class RegistryError(Exception):
    """Base exception for registry errors."""


class AlreadyRegisteredException(RegistryError):
    """A task has already been registered with this name."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Task '{name}' is already registered.")


class NotRegisteredException(RegistryError):
    """No task is registered with this name."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"No task registered with name '{name}'.")


# ── Function-based registration ───────────────────────────────────────────


def register_task(
    name: str,
    *,
    display_name: str = "",
    queue: str = "default",
    parameters: Optional[Type[pydantic.BaseModel]] = None,
    outputs: Optional[Type] = None,
) -> Callable:
    """Decorator that registers a function as a task.

    Tasks are pure computational units with no knowledge of DAG
    topology.  Use :class:`~muflow.pipeline.Pipeline` to compose
    tasks into multi-step DAGs.

    Parameters
    ----------
    name : str
        Unique task identifier (e.g., "myapp.analyse").
    display_name : str, optional
        Human-readable name for UIs.
    queue : str, optional
        Queue name for backend routing. Default: "default".
    parameters : type[pydantic.BaseModel], optional
        Pydantic model for parameter validation.
    outputs : type, optional
        Class with ``files`` dict for output validation.

    Example
    -------
    >>> from typing import Annotated
    >>> import pydantic
    >>> from muflow import register_task, IdentityKey
    >>>
    >>> class MyParams(pydantic.BaseModel):
    ...     id: Annotated[str, IdentityKey()]
    ...     other: str
    >>>
    >>> @register_task("myapp.greet", parameters=MyParams)
    ... def greet(context):
    ...     pass
    """

    def decorator(fn: Callable) -> Callable:
        # Extract identity keys from IdentityKey annotations in parameters model
        final_identity_keys = None
        if parameters is not None:
            final_identity_keys = []
            for field_name, field_info in parameters.model_fields.items():
                for metadata in getattr(field_info, "metadata", []):
                    if isinstance(metadata, IdentityKey):
                        final_identity_keys.append(field_name)
                        break
            if not final_identity_keys:
                final_identity_keys = None

        entry = TaskEntry(
            name=name,
            fn=fn,
            display_name=display_name,
            queue=queue,
            parameters=parameters,
            outputs=outputs,
            identity_keys=final_identity_keys,
        )
        _register_entry(entry)
        return fn

    return decorator


# ── Internal helpers ───────────────────────────────────────────────────────


def _register_entry(entry: TaskEntry) -> None:
    """Store a TaskEntry in the registry."""
    if entry.name in _entries_by_name:
        raise AlreadyRegisteredException(entry.name)
    _entries_by_name[entry.name] = entry
    if entry.display_name:
        _entries_by_display_name[entry.display_name] = entry


# ── Lookup ─────────────────────────────────────────────────────────────────


def get(name: str) -> Optional[TaskEntry]:
    """Get a registered task by name."""
    return _entries_by_name.get(name)


def get_by_display_name(display_name: str) -> Optional[TaskEntry]:
    """Get a registered task by display name."""
    return _entries_by_display_name.get(display_name)


def get_all() -> Dict[str, TaskEntry]:
    """Get all registered tasks."""
    return dict(_entries_by_name)


def get_names() -> list:
    """Get list of all registered task names."""
    return list(_entries_by_name.keys())


def clear() -> None:
    """Clear all registered tasks.  Primarily for testing."""
    _entries_by_name.clear()
    _entries_by_display_name.clear()


def unregister(name: str) -> None:
    """Unregister a task by name.

    Raises
    ------
    NotRegisteredException
        If no task with this name is registered.
    """
    if name not in _entries_by_name:
        raise NotRegisteredException(name)

    entry = _entries_by_name.pop(name)
    if entry.display_name and entry.display_name in _entries_by_display_name:
        del _entries_by_display_name[entry.display_name]
