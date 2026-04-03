"""Explicit pipeline definitions for multi-step workflows.

A Pipeline declares the full DAG topology in one place, keeping individual
workflows as pure computational units with no knowledge of their position
in the DAG.

Example
-------
>>> from muflow.pipeline import Pipeline, Step, ForEach
>>>
>>> pipeline = Pipeline(
...     name="ml.pipeline",
...     display_name="ML Pipeline",
...     steps={
...         "features": ForEach(
...             workflow="ml.compute_features",
...             over=lambda sk, kw: [{"dataset": d} for d in kw["datasets"]],
...         ),
...         "train": Step(workflow="ml.train", after=["features"]),
...         "reports": ForEach(
...             workflow="ml.report",
...             after=["train"],
...             over=lambda sk, kw: [{"format": f} for f in ("pdf", "csv")],
...         ),
...     },
... )
>>>
>>> plan = pipeline.build_plan("tag:1", {"datasets": ["a", "b"]})
>>> print(f"Total nodes: {len(plan.nodes)}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from muflow.plan import WorkflowNode, WorkflowPlan
from muflow.registry import get as get_entry
from muflow.storage import compute_prefix

_log = logging.getLogger(__name__)

# Type for the ``over`` callable: (subject_key, kwargs) -> list[dict]
OverFunc = Callable[[str, dict], List[dict]]


@dataclass
class Step:
    """A single-job step in a pipeline.

    Parameters
    ----------
    workflow : str
        Name of the registered workflow to execute.
    after : list[str]
        Names of steps that must complete before this one.
    kwargs_map : callable, optional
        Function ``(subject_key, kwargs) -> dict`` that computes
        step-specific kwargs from the pipeline kwargs.
    """

    workflow: str
    after: list[str] = field(default_factory=list)
    kwargs_map: Optional[Callable[[str, dict], dict]] = None


@dataclass
class ForEach:
    """A fan-out step that spawns one job per item returned by ``over``.

    Parameters
    ----------
    workflow : str
        Name of the registered workflow to execute (once per item).
    over : callable
        Function ``(subject_key, kwargs) -> list[dict]`` returning
        per-job kwargs.  Each dict is merged with the pipeline kwargs.
    after : list[str]
        Names of steps that must complete before this one.
    """

    workflow: str
    over: OverFunc
    after: list[str] = field(default_factory=list)


@dataclass
class Pipeline:
    """Explicit DAG topology for a multi-step workflow.

    Parameters
    ----------
    name : str
        Unique identifier for this pipeline (used as the sentinel
        workflow name when the final step has multiple nodes).
    display_name : str
        Human-readable name.
    steps : dict[str, Step | ForEach]
        Ordered mapping from step names to step definitions.
        Step names must not contain colons (reserved for access key
        indexing, e.g. ``"features:0"``).

    Example
    -------
    >>> pipeline = Pipeline(
    ...     name="ml.pipeline",
    ...     steps={
    ...         "features": ForEach(
    ...             workflow="ml.features",
    ...             over=lambda sk, kw: [{"id": i} for i in range(3)],
    ...         ),
    ...         "train": Step(workflow="ml.train", after=["features"]),
    ...     },
    ... )
    >>> plan = pipeline.build_plan("tag:1", {})
    """

    name: str
    display_name: str = ""
    steps: Dict[str, Union[Step, ForEach]] = field(default_factory=dict)

    def build_plan(
        self,
        subject_key: str,
        kwargs: dict,
        is_cached: Optional[Callable[[str, str, dict], bool]] = None,
        base_prefix: str = "muflow",
    ) -> WorkflowPlan:
        """Compile this pipeline into a :class:`WorkflowPlan`.

        Parameters
        ----------
        subject_key : str
            Subject identifier (e.g. ``"tag:123"``).
        kwargs : dict
            Pipeline-level parameters, merged with per-step kwargs.
        is_cached : callable, optional
            ``(workflow_name, subject_key, kwargs) -> bool``.
            Default: always False (no caching).
        base_prefix : str
            Base prefix for content-addressed storage paths.

        Returns
        -------
        WorkflowPlan
            Complete execution DAG ready for any backend.
        """
        self._validate_step_names()
        is_cached = is_cached or (lambda *_: False)

        nodes: Dict[str, WorkflowNode] = {}
        # step_name -> list of node keys produced by that step
        step_node_keys: Dict[str, List[str]] = {}

        ordered = self._topological_sort()

        for step_name in ordered:
            step = self.steps[step_name]

            # Enumerate jobs for this step
            if isinstance(step, ForEach):
                job_kwargs_list = step.over(subject_key, kwargs)
            elif isinstance(step, Step) and step.kwargs_map:
                job_kwargs_list = [step.kwargs_map(subject_key, kwargs)]
            else:
                job_kwargs_list = [{}]

            # Collect upstream node keys from ``after`` steps
            upstream_keys = []
            for ref in step.after:
                upstream_keys.extend(step_node_keys.get(ref, []))

            # Build dependency access map: access_key -> storage_prefix
            dep_access_map: Dict[str, str] = {}
            for ref in step.after:
                ref_keys = step_node_keys.get(ref, [])
                if len(ref_keys) == 1:
                    dep_access_map[ref] = nodes[ref_keys[0]].storage_prefix
                else:
                    for i, nk in enumerate(ref_keys):
                        dep_access_map[f"{ref}:{i}"] = nodes[nk].storage_prefix

            # Create one node per job
            step_keys: List[str] = []
            for job_kw in job_kwargs_list:
                merged_kwargs = {**kwargs, **job_kw}
                workflow_name = step.workflow

                entry = get_entry(workflow_name)
                identity_keys = entry.identity_keys if entry else None

                hash_dict = {
                    "workflow": workflow_name,
                    "subject": subject_key,
                    **merged_kwargs,
                }
                storage_prefix = compute_prefix(
                    hash_dict,
                    base_prefix=base_prefix,
                    identity_keys=identity_keys,
                )
                node_key = storage_prefix

                cached = is_cached(workflow_name, subject_key, merged_kwargs)

                output_files: List[str] = []
                if entry and entry.outputs and hasattr(entry.outputs, "files"):
                    output_files = list(entry.outputs.files.keys())

                node = WorkflowNode(
                    key=node_key,
                    function=workflow_name,
                    subject_key=subject_key,
                    kwargs=merged_kwargs,
                    storage_prefix=storage_prefix,
                    depends_on=list(upstream_keys),
                    depended_on_by=[],
                    output_files=output_files,
                    cached=cached,
                    dependency_access_map=dict(dep_access_map),
                )
                nodes[node_key] = node
                step_keys.append(node_key)

            step_node_keys[step_name] = step_keys

        # Compute reverse edges
        for node in nodes.values():
            for dep_key in node.depends_on:
                if dep_key in nodes:
                    nodes[dep_key].depended_on_by.append(node.key)

        # Determine root key
        root_key = self._determine_root(
            ordered, step_node_keys, nodes, subject_key, kwargs, base_prefix,
        )

        _log.debug(
            f"Built pipeline plan '{self.name}': {len(nodes)} nodes, "
            f"root={root_key[:50]}..."
        )

        return WorkflowPlan(nodes=nodes, root_key=root_key)

    # ── Internal helpers ──────────────────────────────────────────

    def _validate_step_names(self) -> None:
        """Raise if any step name contains a colon."""
        for name in self.steps:
            if ":" in name:
                raise ValueError(
                    f"Step name '{name}' must not contain ':' "
                    f"(reserved for access key indexing)."
                )

    def _topological_sort(self) -> list[str]:
        """Return step names in dependency order."""
        visited: set[str] = set()
        in_progress: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in in_progress:
                raise ValueError(
                    f"Circular dependency detected in pipeline steps "
                    f"involving '{name}'."
                )
            in_progress.add(name)
            step = self.steps[name]
            for dep in step.after:
                if dep not in self.steps:
                    raise ValueError(
                        f"Step '{name}' references unknown step '{dep}'."
                    )
                visit(dep)
            in_progress.discard(name)
            visited.add(name)
            order.append(name)

        for name in self.steps:
            visit(name)
        return order

    def _determine_root(
        self,
        ordered: list[str],
        step_node_keys: Dict[str, List[str]],
        nodes: Dict[str, WorkflowNode],
        subject_key: str,
        kwargs: dict,
        base_prefix: str,
    ) -> str:
        """Pick or create the root node for the plan.

        If the last step has exactly one node, it is the root.
        Otherwise a sentinel node is created that depends on all
        terminal nodes.
        """
        last_step = ordered[-1]
        last_keys = step_node_keys[last_step]

        if len(last_keys) == 1:
            return last_keys[0]

        # Create sentinel
        sentinel_key = compute_prefix(
            {"workflow": self.name, "subject": subject_key, **kwargs},
            base_prefix=base_prefix,
        )
        sentinel = WorkflowNode(
            key=sentinel_key,
            function=self.name,
            subject_key=subject_key,
            kwargs=kwargs,
            storage_prefix=sentinel_key,
            depends_on=list(last_keys),
            depended_on_by=[],
            cached=False,
            dependency_access_map={},
        )
        nodes[sentinel_key] = sentinel
        for lk in last_keys:
            nodes[lk].depended_on_by.append(sentinel_key)
        return sentinel_key
