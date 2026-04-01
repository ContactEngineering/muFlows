"""Workflow planning: builds execution DAGs from dependency declarations.

The WorkflowPlanner constructs a complete, static execution plan (DAG) before
any computation begins. This allows:

- Upfront validation of the entire workflow graph
- Content-addressed caching (reuse existing results)
- Backend-agnostic execution (same plan runs on Celery, Lambda, local)
- Direct mapping to Celery chords/groups

Example
-------
>>> from muflow.planner import WorkflowPlanner
>>>
>>> planner = WorkflowPlanner()
>>> plan = planner.build_plan(
...     workflow_name="sds_workflows.gpr_training",
...     subject_key="tag:123",
...     kwargs={"dataset": {...}, "features": [...]},
... )
>>>
>>> # Inspect the plan
>>> print(f"Total nodes: {len(plan.nodes)}")
>>> print(f"Ready to start: {[n.function for n in plan.leaf_nodes()]}")
>>>
>>> # Execute (simplified)
>>> for node in plan.leaf_nodes():
...     submit_to_celery(node)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Set

from muflow.dependencies import enumerate_specs
from muflow.plan import WorkflowNode, WorkflowPlan
from muflow.registry import WorkflowEntry
from muflow.registry import get as get_entry
from muflow.storage import compute_prefix

_log = logging.getLogger(__name__)


class WorkflowPlanner:
    """Builds complete execution DAGs from workflow declarations.

    The planner recursively resolves both dependencies (upstream workflows)
    and productions (downstream workflows) to create a static DAG.

    Parameters
    ----------
    is_cached : callable, optional
        Function with signature ``(workflow_name, subject_key, kwargs) -> bool``.
        Returns True if a cached result exists for this workflow.
        Default: always returns False (no caching).
    base_prefix : str
        Base prefix for content-addressed storage paths.
        Default: "muflow".

    Example
    -------
    >>> def check_cache(name, subject, kwargs):
    ...     # Check S3 or database for existing result
    ...     return storage.exists(f"{compute_prefix(...)}/manifest.json")
    >>>
    >>> planner = WorkflowPlanner(is_cached=check_cache)
    >>> plan = planner.build_plan("myworkflow", "tag:1", {"param": 1})
    """

    def __init__(
        self,
        is_cached: Optional[Callable[[str, str, dict], bool]] = None,
        base_prefix: str = "muflow",
    ):
        self._is_cached = is_cached or (lambda *_: False)
        self._base_prefix = base_prefix

    def build_plan(
        self,
        workflow_name: str,
        subject_key: str,
        kwargs: dict,
    ) -> WorkflowPlan:
        """Build a complete execution plan for a workflow.

        Recursively resolves all dependencies and productions to create
        a static DAG that can be executed on any backend.

        Parameters
        ----------
        workflow_name : str
            Name of the root workflow (e.g., "sds_workflows.gpr_training").
        subject_key : str
            Subject identifier (e.g., "tag:123", "surface:456").
        kwargs : dict
            Parameters for the workflow.

        Returns
        -------
        WorkflowPlan
            Complete execution plan with all nodes and edges.

        Raises
        ------
        ValueError
            If a referenced workflow is not registered.
        """
        nodes: Dict[str, WorkflowNode] = {}

        # Resolve the root workflow and all its dependencies/productions
        root_key = self._resolve(workflow_name, subject_key, kwargs, nodes)

        # Compute reverse edges (depended_on_by)
        for node in nodes.values():
            for dep_key in node.depends_on:
                if dep_key in nodes:
                    nodes[dep_key].depended_on_by.append(node.key)

        _log.debug(
            f"Built plan for {workflow_name}: {len(nodes)} nodes, "
            f"root={root_key[:50]}..."
        )

        return WorkflowPlan(nodes=nodes, root_key=root_key)

    def _resolve(
        self,
        workflow_name: str,
        subject_key: str,
        kwargs: dict,
        nodes: Dict[str, WorkflowNode],
        _resolving: Optional[Set[str]] = None,
    ) -> str:
        """Recursively resolve a workflow and its dependencies/productions.

        Parameters
        ----------
        workflow_name : str
            Name of the workflow to resolve.
        subject_key : str
            Subject identifier.
        kwargs : dict
            Workflow parameters.
        nodes : dict
            Accumulator for discovered nodes (mutated).
        _resolving : set, optional
            Set of node keys currently being resolved (for cycle detection).

        Returns
        -------
        str
            The node key for this workflow.
        """
        if _resolving is None:
            _resolving = set()

        # Get workflow entry from registry
        entry = get_entry(workflow_name)
        if entry is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Compute content-addressed storage prefix (also used as node key)
        hash_dict = {"workflow": workflow_name, "subject": subject_key, **kwargs}
        storage_prefix = compute_prefix(
            hash_dict, base_prefix=self._base_prefix, identity_keys=entry.identity_keys
        )
        node_key = storage_prefix

        # Cycle detection - check BEFORE checking nodes to catch cycles
        if node_key in _resolving:
            raise ValueError(
                f"Circular dependency detected: {workflow_name} "
                f"(subject={subject_key}) references itself"
            )

        # Already processed this exact node?
        if node_key in nodes:
            return node_key

        _resolving.add(node_key)

        try:
            # Check cache
            cached = self._is_cached(workflow_name, subject_key, kwargs)

            # Get output files from entry (if declared)
            output_files = []
            if entry.outputs is not None and hasattr(entry.outputs, "files"):
                output_files = list(entry.outputs.files.keys())

            # Create the node
            node = WorkflowNode(
                key=node_key,
                function=workflow_name,
                subject_key=subject_key,
                kwargs=kwargs,
                storage_prefix=storage_prefix,
                depends_on=[],
                depended_on_by=[],
                output_files=output_files,
                cached=cached,
            )
            nodes[node_key] = node

            # Skip dependency/production resolution if cached
            # (cached nodes don't need their deps resolved)
            if cached:
                _log.debug(f"Skipping cached node: {workflow_name}")
                return node_key

            # Resolve dependencies (upstream - must complete before this node)
            self._resolve_dependencies(node, entry, nodes, _resolving)

            # Resolve productions (downstream - spawned after this node completes)
            self._resolve_productions(node, entry, nodes, _resolving)

            return node_key

        finally:
            _resolving.discard(node_key)

    def _resolve_dependencies(
        self,
        node: WorkflowNode,
        entry: WorkflowEntry,
        nodes: Dict[str, WorkflowNode],
        _resolving: Set[str],
    ) -> None:
        """Resolve dependencies for a node.

        Parameters
        ----------
        node : WorkflowNode
            The node to resolve dependencies for.
        entry : WorkflowEntry
            The workflow entry for this node.
        nodes : dict
            Accumulator for discovered nodes (mutated).
        _resolving : set
            Set of node keys currently being resolved.
        """
        dep_specs = enumerate_specs(entry.dependencies, node.subject_key, node.kwargs)
        for dep_key, spec in dep_specs.items():
            dep_node_key = self._resolve(
                spec.workflow,
                spec.subject_key,
                spec.kwargs,
                nodes,
                _resolving,
            )
            node.depends_on.append(dep_node_key)
            _log.debug(f"  {node.function} depends on {spec.workflow} ({dep_key})")

    def _resolve_productions(
        self,
        node: WorkflowNode,
        entry: WorkflowEntry,
        nodes: Dict[str, WorkflowNode],
        _resolving: Set[str],
    ) -> None:
        """Resolve productions for a node.

        Parameters
        ----------
        node : WorkflowNode
            The node to resolve productions for.
        entry : WorkflowEntry
            The workflow entry for this node.
        nodes : dict
            Accumulator for discovered nodes (mutated).
        _resolving : set
            Set of node keys currently being resolved.
        """
        produces = getattr(entry, "produces", None)
        if produces:
            prod_specs = enumerate_specs(produces, node.subject_key, node.kwargs)
            for prod_key, spec in prod_specs.items():
                prod_node_key = self._resolve(
                    spec.workflow,
                    spec.subject_key,
                    spec.kwargs,
                    nodes,
                    _resolving,
                )
                # The produced workflow depends on this node
                nodes[prod_node_key].depends_on.append(node.key)
                _log.debug(f"  {node.function} produces {spec.workflow} ({prod_key})")

    def _compute_dependency_keys(
        self,
        node: WorkflowNode,
        entry: WorkflowEntry,
    ) -> Dict[str, str]:
        """Compute the mapping from dependency access keys to node keys.

        This mapping allows workflows to access dependencies by their
        declared key (e.g., "surface_0") rather than the full node key.

        Parameters
        ----------
        node : WorkflowNode
            The workflow node.
        entry : WorkflowEntry
            The workflow entry with dependency declarations.

        Returns
        -------
        dict[str, str]
            Mapping from access key to node key.
        """
        dep_specs = enumerate_specs(entry.dependencies, node.subject_key, node.kwargs)

        result = {}
        for access_key, spec in dep_specs.items():
            # Compute the node key for this dependency
            hash_dict = {
                "workflow": spec.workflow,
                "subject": spec.subject_key,
                **spec.kwargs,
            }
            # Look up dependency's entry to get its identity_keys
            dep_entry = get_entry(spec.workflow)
            dep_identity_keys = dep_entry.identity_keys if dep_entry else None
            dep_node_key = compute_prefix(
                hash_dict,
                base_prefix=self._base_prefix,
                identity_keys=dep_identity_keys,
            )
            result[access_key] = dep_node_key

        return result


def get_dependency_access_map(
    plan: WorkflowPlan,
    node_key: str,
    base_prefix: str = "muflow",
) -> Dict[str, str]:
    """Get the mapping from dependency access keys to storage prefixes.

    This is used at execution time to set up the WorkflowContext with
    the correct dependency storages.

    Parameters
    ----------
    plan : WorkflowPlan
        The execution plan.
    node_key : str
        Key of the node to get dependencies for.
    base_prefix : str
        Base prefix used for storage paths.

    Returns
    -------
    dict[str, str]
        Mapping from access key (e.g., "surface_0") to storage prefix.

    Note
    ----
    This function re-enumerates dependencies to recover the access keys.
    The plan's ``get_dependency_prefixes`` method returns node keys as keys,
    but workflows access dependencies by their declared access key.
    """
    node = plan.nodes[node_key]
    entry = get_entry(node.function)

    if entry is None:
        return {}

    dep_specs = enumerate_specs(entry.dependencies, node.subject_key, node.kwargs)

    result = {}
    for access_key, spec in dep_specs.items():
        # Compute the storage prefix for this dependency
        hash_dict = {
            "workflow": spec.workflow,
            "subject": spec.subject_key,
            **spec.kwargs,
        }
        storage_prefix = compute_prefix(hash_dict, base_prefix=base_prefix)
        result[access_key] = storage_prefix

    return result
