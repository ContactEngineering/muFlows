"""Basic workflow registry.

This module provides a simple registry for workflow implementations.
Workflows are registered by their Meta.name attribute.

Example
-------
>>> from muflow import WorkflowImplementation
>>> from muflow.registry import register, get, get_all
>>>
>>> @register
... class MyWorkflow(WorkflowImplementation):
...     class Meta:
...         name = "myapp.my_workflow"
...         display_name = "My Workflow"
...
...     def execute(self, context):
...         return {"result": "done"}
>>>
>>> # Retrieve by name
>>> workflow_class = get("myapp.my_workflow")
>>> workflow = workflow_class()
"""

from typing import Dict, Optional, Type

# Registry storage
_implementations_by_name: Dict[str, Type] = {}
_implementations_by_display_name: Dict[str, Type] = {}


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


def register(klass: Type) -> Type:
    """Register a workflow implementation class.

    Can be used as a decorator:

        @register
        class MyWorkflow(WorkflowImplementation):
            ...

    Or called directly:

        register(MyWorkflow)

    Parameters
    ----------
    klass : Type
        The workflow implementation class to register.

    Returns
    -------
    Type
        The registered class (allows use as decorator).

    Raises
    ------
    AlreadyRegisteredException
        If a workflow with the same name is already registered.
    ValueError
        If the class has no Meta.name attribute.
    """
    name = getattr(klass.Meta, "name", None)
    if not name:
        raise ValueError(
            f"Workflow class {klass.__name__} has no Meta.name attribute."
        )

    if name in _implementations_by_name:
        raise AlreadyRegisteredException(name)

    _implementations_by_name[name] = klass

    display_name = getattr(klass.Meta, "display_name", None)
    if display_name:
        _implementations_by_display_name[display_name] = klass

    return klass


def get(name: str) -> Optional[Type]:
    """Get a registered workflow by name.

    Parameters
    ----------
    name : str
        The workflow name (Meta.name).

    Returns
    -------
    Type or None
        The workflow class, or None if not found.
    """
    return _implementations_by_name.get(name)


def get_by_display_name(display_name: str) -> Optional[Type]:
    """Get a registered workflow by display name.

    Parameters
    ----------
    display_name : str
        The workflow display name (Meta.display_name).

    Returns
    -------
    Type or None
        The workflow class, or None if not found.
    """
    return _implementations_by_display_name.get(display_name)


def get_all() -> Dict[str, Type]:
    """Get all registered workflows.

    Returns
    -------
    dict
        Dictionary mapping workflow names to classes.
    """
    return dict(_implementations_by_name)


def get_names() -> list:
    """Get list of all registered workflow names.

    Returns
    -------
    list
        List of workflow names.
    """
    return list(_implementations_by_name.keys())


def clear() -> None:
    """Clear all registered workflows.

    Primarily for testing.
    """
    _implementations_by_name.clear()
    _implementations_by_display_name.clear()


def unregister(name: str) -> None:
    """Unregister a workflow by name.

    Parameters
    ----------
    name : str
        The workflow name to unregister.

    Raises
    ------
    NotRegisteredException
        If no workflow with this name is registered.
    """
    if name not in _implementations_by_name:
        raise NotRegisteredException(name)

    klass = _implementations_by_name.pop(name)

    display_name = getattr(klass.Meta, "display_name", None)
    if display_name and display_name in _implementations_by_display_name:
        del _implementations_by_display_name[display_name]
