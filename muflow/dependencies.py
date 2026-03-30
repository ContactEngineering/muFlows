"""Workflow dependency and production specifications.

This module provides data structures for declaring workflow dependencies
(upstream workflows that must complete first) and productions (downstream
workflows that are spawned after completion).

Both dependencies and productions can be:
- Static: A fixed workflow name or WorkflowSpec
- Dynamic: A callable that enumerates specs at plan time

Example
-------
>>> from muflow.dependencies import WorkflowSpec
>>>
>>> # Static dependency
>>> dependencies = {"features": "myapp.compute_features"}
>>>
>>> # Explicit spec with custom kwargs
>>> dependencies = {
...     "features": WorkflowSpec(
...         workflow="myapp.compute_features",
...         kwargs={"resolution": "high"},
...     )
... }
>>>
>>> # Dynamic enumeration (one dependency per surface)
>>> def enumerate_surfaces(subject_key, kwargs):
...     surfaces = kwargs.get("surfaces", [])
...     return {
...         f"surface_{i}": WorkflowSpec(
...             workflow="myapp.process_surface",
...             subject_key=f"surface:{s['id']}",
...         )
...         for i, s in enumerate(surfaces)
...     }
>>> dependencies = enumerate_surfaces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union


@dataclass
class WorkflowSpec:
    """Specification for a dependent or produced workflow.

    Used to declare relationships between workflows in the execution DAG.
    Can specify the target workflow, subject, and parameters.

    Attributes
    ----------
    workflow : str
        Name of the workflow (e.g., "sds_workflows.feature_vector").
    subject_key : str, optional
        Subject key for the workflow. If None, inherits from parent workflow.
    kwargs : dict, optional
        Parameters for the workflow. If None, inherits from parent workflow.
    key : str, optional
        Access key for ``context.dependency(key)``. Auto-generated from
        the declaration key if not specified.

    Example
    -------
    >>> spec = WorkflowSpec(
    ...     workflow="sds_workflows.feature_vector",
    ...     subject_key="surface:123",
    ...     kwargs={"features": ["height", "slope"]},
    ... )
    """

    workflow: str
    subject_key: Optional[str] = None
    kwargs: Optional[dict] = None
    key: Optional[str] = None

    def with_defaults(
        self,
        default_subject_key: str,
        default_kwargs: dict,
    ) -> "WorkflowSpec":
        """Return a new spec with defaults applied.

        Parameters
        ----------
        default_subject_key : str
            Subject key to use if this spec has None.
        default_kwargs : dict
            Kwargs to use if this spec has None.

        Returns
        -------
        WorkflowSpec
            New spec with defaults applied.
        """
        return WorkflowSpec(
            workflow=self.workflow,
            subject_key=self.subject_key if self.subject_key is not None else default_subject_key,
            kwargs=self.kwargs if self.kwargs is not None else default_kwargs,
            key=self.key,
        )


# Type alias for enumeration functions
# Signature: (subject_key: str, kwargs: dict) -> Dict[str, WorkflowSpec]
EnumerationFunc = Callable[[str, dict], Dict[str, WorkflowSpec]]

# A declaration can be:
# - str: Simple workflow name (inherits subject and kwargs from parent)
# - WorkflowSpec: Explicit specification
# - EnumerationFunc: Dynamic enumeration at plan time
WorkflowDeclaration = Union[str, WorkflowSpec, EnumerationFunc]

# Full dependencies/produces declaration
# Can be a dict mapping keys to declarations, or a single enumeration function
DependencyDict = Union[Dict[str, WorkflowDeclaration], EnumerationFunc, None]


def enumerate_specs(
    declaration: DependencyDict,
    subject_key: str,
    kwargs: dict,
) -> Dict[str, WorkflowSpec]:
    """Enumerate all workflow specs from a declaration.

    Handles all declaration types: dicts, callables, strings, and WorkflowSpecs.

    Parameters
    ----------
    declaration : dict, callable, or None
        The dependency or production declaration from a WorkflowEntry.
    subject_key : str
        Subject key of the parent workflow.
    kwargs : dict
        Parameters of the parent workflow.

    Returns
    -------
    dict[str, WorkflowSpec]
        Mapping from access key to resolved WorkflowSpec.

    Example
    -------
    >>> decl = {"features": "myapp.features", "config": WorkflowSpec(...)}
    >>> specs = enumerate_specs(decl, "tag:1", {"param": "value"})
    >>> specs["features"].workflow
    'myapp.features'
    """
    if declaration is None:
        return {}

    if callable(declaration):
        # Single enumeration function that returns all specs
        result = declaration(subject_key, kwargs)
        # Apply defaults to all returned specs
        return {
            key: spec.with_defaults(subject_key, kwargs)
            for key, spec in result.items()
        }

    if isinstance(declaration, dict):
        result = {}
        for key, spec in declaration.items():
            if callable(spec):
                # Nested enumeration for this key
                sub_specs = spec(subject_key, kwargs)
                for sub_key, sub_spec in sub_specs.items():
                    full_key = f"{key}.{sub_key}" if key else sub_key
                    result[full_key] = sub_spec.with_defaults(subject_key, kwargs)
            elif isinstance(spec, str):
                # Simple workflow name - inherit everything
                result[key] = WorkflowSpec(
                    workflow=spec,
                    subject_key=subject_key,
                    kwargs=kwargs,
                    key=key,
                )
            elif isinstance(spec, WorkflowSpec):
                result[key] = spec.with_defaults(subject_key, kwargs)
            else:
                raise TypeError(
                    f"Invalid dependency spec for key '{key}': {type(spec).__name__}. "
                    f"Expected str, WorkflowSpec, or callable."
                )
        return result

    raise TypeError(
        f"Invalid declaration type: {type(declaration).__name__}. "
        f"Expected dict, callable, or None."
    )
