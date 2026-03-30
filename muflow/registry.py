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
>>> from muflow import WorkflowImplementation
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
from typing import Any, Callable, Dict, Optional, Type, Union

import pydantic


# ── WorkflowEntry ──────────────────────────────────────────────────────────

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
    """

    name: str
    fn: Callable
    display_name: str = ""
    queue: str = "default"
    dependencies: Union[dict, Callable, None] = field(default_factory=dict)
    produces: Union[dict, Callable, None] = field(default_factory=dict)
    parameters: Optional[Type[pydantic.BaseModel]] = None
    outputs: Optional[Type] = None


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
    >>> @register_workflow(
    ...     name="myapp.analyse",
    ...     display_name="Analyse",
    ...     dependencies={"features": "myapp.features"},
    ...     parameters=MyParams,
    ...     outputs=MyOutputs,
    ... )
    ... def analyse(context):
    ...     features = context.dependency("features").read_json("result.json")
    ...     ...
    """
    def decorator(fn: Callable) -> Callable:
        entry = WorkflowEntry(
            name=name,
            fn=fn,
            display_name=display_name,
            queue=queue,
            dependencies=dependencies or {},
            produces=produces or {},
            parameters=parameters,
            outputs=outputs,
        )
        _register_entry(entry)
        return fn
    return decorator


# ── Class-based registration (legacy) ──────────────────────────────────────

def register(klass: Type) -> Type:
    """Register a ``WorkflowImplementation`` subclass.

    Can be used as a decorator or called directly.  Internally wraps the
    class in a ``WorkflowEntry``.

    Parameters
    ----------
    klass : Type
        A ``WorkflowImplementation`` subclass with a ``Meta.name`` attribute.

    Returns
    -------
    Type
        The registered class (unchanged).
    """
    meta_name = getattr(getattr(klass, "Meta", None), "name", None)
    if not meta_name:
        raise ValueError(
            f"Workflow class {klass.__name__} has no Meta.name attribute."
        )

    meta = klass.Meta
    display_name = getattr(meta, "display_name", "")
    queue = getattr(meta, "queue", "default")
    dependencies = getattr(meta, "dependencies", {})
    produces = getattr(meta, "produces", {})
    parameters_cls = getattr(klass, "Parameters", None)
    # Only keep Parameters if it's a subclass that adds fields
    if parameters_cls is not None:
        try:
            if not parameters_cls.model_fields:
                parameters_cls = None
        except AttributeError:
            parameters_cls = None

    def _class_wrapper(context):
        """Wrapper that instantiates the class and calls execute()."""
        if context.parameters is not None:
            impl = klass(**context.parameters.model_dump())
        else:
            impl = klass()
        return impl.execute(context)

    entry = WorkflowEntry(
        name=meta_name,
        fn=_class_wrapper,
        display_name=display_name,
        queue=queue,
        dependencies=dependencies,
        produces=produces,
        parameters=parameters_cls,
    )
    # Store the original class on the entry for backward compatibility
    entry._class = klass  # noqa: SLF001
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
