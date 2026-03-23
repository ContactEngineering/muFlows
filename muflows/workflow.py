"""Base class for workflow implementations.

WorkflowImplementation provides the abstract base for defining workflows that
can run on any backend (Celery, Lambda, Batch). Concrete implementations
override the execute() method to perform their computation.

This module is backend-agnostic and has no Django dependencies.
"""

from typing import Union

import pydantic

from .context import WorkflowContext


class WorkflowImplementation:
    """Base class for workflow implementations.

    Subclasses should:
    1. Define a Parameters class for validated kwargs
    2. Implement execute() method for the new context-based interface

    The Meta class provides workflow metadata:
    - name: Unique identifier (e.g., "topobank_statistics.height_distribution")
    - display_name: Human-readable name (e.g., "Height distribution")

    Note: execute() is not abstract to allow incremental migration from legacy
    implementations (topography_implementation, surface_implementation, etc.).
    Once all workflows are migrated, execute() may become abstract.

    Example
    -------
    class MyWorkflow(WorkflowImplementation):
        class Meta:
            name = "myapp.my_workflow"
            display_name = "My Workflow"

        class Parameters(WorkflowImplementation.Parameters):
            threshold: float = 0.5
            iterations: int = 100

        def execute(self, context: WorkflowContext) -> dict:
            threshold = context.kwargs.get("threshold", 0.5)
            iterations = context.kwargs.get("iterations", 100)

            # Perform computation...
            result = {"accuracy": 0.95}

            # Save outputs
            context.save_json("result.json", result)

            return result
    """

    class Meta:
        """Workflow metadata. Override in subclasses."""

        name: str = ""
        display_name: str = ""

    class Parameters(pydantic.BaseModel):
        """Workflow parameters schema. Override in subclasses."""

        model_config = pydantic.ConfigDict(extra="forbid")

    def __init__(self, **kwargs):
        """Initialize with validated parameters.

        Parameters are validated against the Parameters schema and stored
        for access via self.kwargs. For the new execute() interface, kwargs
        should be accessed via context.kwargs instead.
        """
        self._kwargs = self.Parameters(**kwargs)

    @property
    def kwargs(self):
        """Return the validated parameters object."""
        return self._kwargs

    def execute(self, context: WorkflowContext) -> dict:
        """Execute the workflow.

        Subclasses should override this method to implement the workflow logic.
        The default implementation raises NotImplementedError.

        Parameters
        ----------
        context : WorkflowContext
            Provides kwargs, file I/O, dependencies, and progress reporting.

        Returns
        -------
        dict
            Result data. Typically saved as result.json by the caller.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement execute(). "
            "Override execute() to use the context-based interface."
        )

    @classmethod
    def clean_kwargs(cls, kwargs: Union[dict, None], fill_missing: bool = True) -> dict:
        """Validate keyword arguments and return cleaned dictionary.

        Parameters
        ----------
        kwargs : dict or None
            Keyword arguments to validate.
        fill_missing : bool, optional
            If True, fill missing keys with default values. Default is True.

        Returns
        -------
        dict
            Validated and cleaned kwargs.

        Raises
        ------
        pydantic.ValidationError
            If validation fails.
        """
        if kwargs is None:
            if fill_missing:
                return cls.Parameters().model_dump()
            else:
                return {}
        return cls.Parameters(**kwargs).model_dump(exclude_unset=not fill_missing)
