"""Workflow registry.

Workflows can be registered as plain functions (preferred) or as
``WorkflowImplementation`` subclasses (legacy).  Both are stored as
``WorkflowEntry`` objects.

Function-based registration
---------------------------
>>> from muflow.registry import register_workflow
>>> import pydantic
>>>
>>> class MyParams(pydantic.BaseModel):
...     threshold: float = 0.5
>>>
>>> @register_workflow(
...     name="myapp.my_workflow",
...     display_name="My Workflow",
...     parameters=MyParams,
... )
... def my_workflow(context):
...     return {"result": "done"}

Class-based registration (legacy)
---------------------------------
>>> from topobank.analysis.workflows import WorkflowImplementation
>>> from muflow.registry import register
>>>
>>> @register
... class MyWorkflow(WorkflowImplementation):
...     class Meta:
...         name = "myapp.my_workflow"
...         display_name = "My Workflow"
...     def execute(self, context):
...         return {"result": "done"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Union

import pydantic

# ── WorkflowEntry ──────────────────────────────────────────────────────────


@dataclass
class IdentityKey:
    """Marker for Pydantic model fields that define the workflow's identity."""
    pass


@dataclass
class WorkflowEntry:
    """Unified descriptor for a registered workflow.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. ``"topobank_statistics.autocorrelation"``).
    fn : Callable
        The workflow function.  Signature: ``fn(context) -> dict | None``.
    display_name : str
        Human-readable name shown in UIs.
    queue : str
        Queue name for backend routing.
    dependencies : dict | callable | None
        Upstream workflows that must complete before this one.
        Can be a dict mapping access keys to workflow names/specs,
        or a callable that enumerates dependencies at plan time.
    produces : dict | callable | None
        Downstream workflows spawned after this one completes (fan-out).
        Same format as dependencies.
    parameters : type[pydantic.BaseModel] | None
        Pydantic model for parameter validation.  ``None`` means no
        parameters.
    outputs : type | None
        An inner ``Outputs`` class with a ``files`` dict mapping filenames
        to ``OutputFile`` descriptors.  Used for output validation.
    identity_keys : list[str] | None
        List of keys in kwargs that define the workflow's identity for hashing.
        If None, all kwargs are used.
    """

    name: str
    fn: Callable
    display_name: str = ""
    queue: str = "default"
    dependencies: Union[dict, Callable, None] = field(default_factory=dict)
    produces: Union[dict, Callable, None] = field(default_factory=dict)
    parameters: Optional[Type[pydantic.BaseModel]] = None
    outputs: Optional[Type] = None
    identity_keys: Optional[List[str]] = None


# ── Registry storage ───────────────────────────────────────────────────────

_entries_by_name: Dict[str, WorkflowEntry] = {}
_entries_by_display_name: Dict[str, WorkflowEntry] = {}


# ── Exceptions ─────────────────────────────────────────────────────────────


class RegistryError(Exception):
    """Base exception for registry errors."""


class AlreadyRegisteredException(RegistryError):
    """A workflow has already been registered with this name."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Workflow '{name}' is already registered.")


class NotRegisteredException(RegistryError):
    """No workflow is registered with this name."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"No workflow registered with name '{name}'.")


# ── Function-based registration (preferred) ────────────────────────────────


def register_workflow(
    name: str,
    *,
    display_name: str = "",
    queue: str = "default",
    dependencies: Union[dict, Callable] = None,
    produces: Union[dict, Callable] = None,
    parameters: Optional[Type[pydantic.BaseModel]] = None,
    outputs: Optional[Type] = None,
) -> Callable:
    """Decorator that registers a function as a workflow.

    Parameters
    ----------
    name : str
        Unique workflow identifier (e.g., "myapp.analyse").
    display_name : str, optional
        Human-readable name for UIs.
    queue : str, optional
        Queue name for backend routing. Default: "default".
    dependencies : dict or callable, optional
        Upstream workflows that must complete before this one.
        Can be a dict mapping access keys to workflow names/specs,
        or a callable ``(subject_key, kwargs) -> Dict[str, WorkflowSpec]``.
    produces : dict or callable, optional
        Downstream workflows spawned after this one completes (fan-out).
        Same format as dependencies.
    parameters : type[pydantic.BaseModel], optional
        Pydantic model for parameter validation.
    outputs : type, optional
        Class with ``files`` dict for output validation.

    Example
    -------
    >>> from typing import Annotated
    >>> import pydantic
    >>> from muflow import register_workflow, IdentityKey
    >>>
    >>> class MyParams(pydantic.BaseModel):
    ...     id: Annotated[str, IdentityKey()]
    ...     other: str
    >>>
    >>> @register_workflow("myapp.greet", parameters=MyParams)
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

        entry = WorkflowEntry(
            name=name,
            fn=fn,
            display_name=display_name,
            queue=queue,
            dependencies=dependencies or {},
            produces=produces or {},
            parameters=parameters,
            outputs=outputs,
            identity_keys=final_identity_keys,
        )
        _register_entry(entry)
        return fn

    return decorator


# ── Legacy registration (class-based) ──────────────────────────────────


def register(klass: Type) -> Type:
    """Decorator for class-based workflow registration (legacy).

    Used for ``WorkflowImplementation`` subclasses.
    """
    # Create an entry from the class Meta
    meta = getattr(klass, "Meta", None)
    if meta is None:
        raise RegistryError(f"Class {klass.__name__} must have a Meta inner class.")

    # Create the function wrapper for the class-based execute()
    def wrapper(context):
        obj = klass()
        return obj.execute(context)

    parameters = getattr(meta, "parameters", None)
    legacy_identity_keys = None
    if parameters is not None:
        legacy_identity_keys = []
        for field_name, field_info in parameters.model_fields.items():
            for metadata in getattr(field_info, "metadata", []):
                if isinstance(metadata, IdentityKey):
                    legacy_identity_keys.append(field_name)
                    break
        if not legacy_identity_keys:
            legacy_identity_keys = None

    entry = WorkflowEntry(
        name=getattr(meta, "name"),
        fn=wrapper,
        display_name=getattr(meta, "display_name", ""),
        queue=getattr(meta, "queue", "default"),
        dependencies=getattr(meta, "dependencies", {}),
        produces=getattr(meta, "produces", {}),
        parameters=parameters,
        outputs=getattr(klass, "Outputs", None),
        identity_keys=legacy_identity_keys,
    )
    _register_entry(entry)
    return klass


# ── Internal helpers ───────────────────────────────────────────────────────


def _register_entry(entry: WorkflowEntry) -> None:
    """Store a WorkflowEntry in the registry."""
    if entry.name in _entries_by_name:
        raise AlreadyRegisteredException(entry.name)
    _entries_by_name[entry.name] = entry
    if entry.display_name:
        _entries_by_display_name[entry.display_name] = entry


# ── Lookup ─────────────────────────────────────────────────────────────────


def get(name: str) -> Optional[WorkflowEntry]:
    """Get a registered workflow by name.

    Parameters
    ----------
    name : str
        The workflow name.

    Returns
    -------
    WorkflowEntry or None
        The workflow entry, or None if not found.
    """
    return _entries_by_name.get(name)


def get_by_display_name(display_name: str) -> Optional[WorkflowEntry]:
    """Get a registered workflow by display name.

    Parameters
    ----------
    display_name : str
        The workflow display name.

    Returns
    -------
    WorkflowEntry or None
        The workflow entry, or None if not found.
    """
    return _entries_by_display_name.get(display_name)


def get_all() -> Dict[str, WorkflowEntry]:
    """Get all registered workflows.

    Returns
    -------
    dict
        Dictionary mapping workflow names to ``WorkflowEntry`` objects.
    """
    return dict(_entries_by_name)


def get_names() -> list:
    """Get list of all registered workflow names.

    Returns
    -------
    list
        List of workflow names.
    """
    return list(_entries_by_name.keys())


def clear() -> None:
    """Clear all registered workflows.  Primarily for testing."""
    _entries_by_name.clear()
    _entries_by_display_name.clear()


def unregister(name: str) -> None:
    """Unregister a workflow by name.

    Raises
    ------
    NotRegisteredException
        If no workflow with this name is registered.
    """
    if name not in _entries_by_name:
        raise NotRegisteredException(name)

    entry = _entries_by_name.pop(name)
    if entry.display_name and entry.display_name in _entries_by_display_name:
        del _entries_by_display_name[entry.display_name]
