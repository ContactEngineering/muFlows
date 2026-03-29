"""Mixin for workflow parameter support.

The base ``WorkflowContext`` protocol is agnostic of workflow parameters.
Contexts that need to carry validated parameters (most do) use this mixin.
"""

from __future__ import annotations

from typing import Any


class ParameterizedMixin:
    """Mixin that adds ``kwargs`` and ``parameters`` to a context.

    Provides two properties:

    - ``kwargs`` — the raw parameter dict.
    - ``parameters`` — the validated pydantic model (set by the executor
      after validation, or ``None`` if no parameter model is registered).

    Classes using this mixin must initialise ``_kwargs`` and ``_parameters``
    in their ``__init__``.
    """

    _kwargs: dict
    _parameters: Any

    @property
    def kwargs(self) -> dict:
        """Raw parameters dict."""
        return self._kwargs

    @property
    def parameters(self) -> Any:
        """Validated parameters (pydantic model), or ``None``."""
        return self._parameters
