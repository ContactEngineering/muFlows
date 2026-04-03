"""Testing utilities for muFlow.

This module provides helper functions for running workflows in tests
without manual dependency management.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from muflow.plan import WorkflowPlan
    from muflow.pipeline import Pipeline

_log = logging.getLogger(__name__)


@dataclass
class LocalExecutionResult:
    """Result of a local workflow execution.

    Attributes
    ----------
    success : bool
        Whether all nodes executed successfully.
    plan : WorkflowPlan
        The executed plan.
    output_dir : Path
        Directory containing all outputs.
    root_output_dir : Path
        Directory containing root workflow outputs.
    error : str | None
        Error message if execution failed.
    """

    success: bool
    plan: "WorkflowPlan"
    output_dir: Path
    root_output_dir: Path
    error: Optional[str] = None

    def read_json(self, filename: str) -> Any:
        """Read a JSON file from the root workflow output.

        Parameters
        ----------
        filename : str
            Name of the JSON file (e.g., "training_result.json").

        Returns
        -------
        Any
            Parsed JSON content.
        """
        path = self.root_output_dir / filename
        with open(path) as f:
            return json.load(f)

    def read_file(self, filename: str) -> bytes:
        """Read a file from the root workflow output.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        bytes
            File contents.
        """
        path = self.root_output_dir / filename
        return path.read_bytes()

    def list_files(self) -> list[str]:
        """List files in the root workflow output directory.

        Returns
        -------
        list[str]
            List of filenames.
        """
        return [f.name for f in self.root_output_dir.iterdir() if f.is_file()]


def run_plan_locally(
    pipeline: "Pipeline",
    subject_key: str,
    kwargs: dict,
    output_dir: Union[str, Path],
    verbose: bool = False,
    use_cache: bool = True,
) -> LocalExecutionResult:
    """Run a pipeline with all steps computed automatically.

    This is a convenience function for testing that:
    1. Builds a complete execution plan from a Pipeline
    2. Executes all nodes using LocalBackend
    3. Returns the result with easy access to outputs

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to execute.
    subject_key : str
        Subject identifier (e.g., "tag:123", "dataset:test").
    kwargs : dict
        Pipeline parameters.
    output_dir : str or Path
        Directory for all workflow outputs.
    verbose : bool
        If True, print progress messages.
    use_cache : bool
        If True (default), skip execution of workflows that have cached results.

    Returns
    -------
    LocalExecutionResult
        Execution result with access to outputs.

    Example
    -------
    >>> from muflow.testing import run_plan_locally
    >>> from muflow.examples.ml_workflow import ml_pipeline
    >>>
    >>> result = run_plan_locally(
    ...     pipeline=ml_pipeline,
    ...     subject_key="experiment:1",
    ...     kwargs={"datasets": ["a", "b", "c"]},
    ...     output_dir="/tmp/test_output",
    ... )
    >>>
    >>> if result.success:
    ...     print(f"Plan has {len(result.plan.nodes)} nodes")
    """
    from muflow.backends import LocalBackend
    from muflow.storage import LocalStorageBackend

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the plan
    if verbose:
        print(f"Planning {pipeline.name}...")

    base_path = str(output_dir.absolute())
    is_cached = LocalStorageBackend.make_cache_checker(base_path) if use_cache else None
    plan = pipeline.build_plan(
        subject_key=subject_key,
        kwargs=kwargs,
        is_cached=is_cached,
        base_prefix=base_path,
    )

    if verbose:
        print(f"Plan has {len(plan.nodes)} nodes")

    # Execute
    if verbose:
        print("Executing plan...")

    backend = LocalBackend(base_path=str(output_dir.absolute()))

    def on_complete(node_key: str):
        if verbose:
            print(f"  [DONE] {node_key[:50]}...")

    def on_failure(node_key: str, error: str):
        if verbose:
            print(f"  [FAIL] {node_key[:50]}...: {error}")

    try:
        backend.submit_plan(
            plan,
            on_node_complete=on_complete,
            on_node_failure=on_failure,
        )
        success = True
        error = None
    except RuntimeError as e:
        success = False
        error = str(e)

    # Get root output directory
    root_node = plan.nodes[plan.root_key]
    root_output_dir = Path(root_node.storage_prefix)

    return LocalExecutionResult(
        success=success,
        plan=plan,
        output_dir=output_dir,
        root_output_dir=root_output_dir,
        error=error,
    )
