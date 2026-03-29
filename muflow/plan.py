"""Workflow plan data structures.

A WorkflowPlan represents a complete, static execution DAG. It is computed
once upfront and stored as JSON. The plan contains all information needed
to execute the workflow on any backend.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import pydantic


class WorkflowNode(pydantic.BaseModel):
    """A single node in the execution plan.

    Attributes
    ----------
    key : str
        Unique identifier within the plan. Typically derived from
        function name, subject, and kwargs.
    function : str
        Workflow function name (e.g., "sds_ml.v3.gpr.training").
    subject_key : str
        Identifier for the subject (e.g., S3 key or "tag:123").
    kwargs : dict
        Parameters passed to the workflow.
    storage_prefix : str
        Content-addressed S3 key prefix for output files.
    depends_on : list[str]
        Keys of upstream nodes that must complete first.
    depended_on_by : list[str]
        Keys of downstream nodes waiting on this one.
    output_files : list[str]
        Filenames this node will produce (from Outputs schema).
    cached : bool
        True if results already exist at storage_prefix.
    analysis_id : int, optional
        Database ID of the WorkflowResult (set after DB record creation).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    key: str
    function: str
    subject_key: str
    kwargs: dict
    storage_prefix: str
    depends_on: list[str] = []
    depended_on_by: list[str] = []
    output_files: list[str] = []
    cached: bool = False
    analysis_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowNode:
        """Create from dictionary."""
        return cls.model_validate(data)


class WorkflowPlan(pydantic.BaseModel):
    """A complete, static execution DAG.

    The plan is computed once by the WorkflowPlanner and stored as JSON.
    It contains all information needed to:
    - Create write-ahead database records
    - Submit nodes to execution backends
    - Track completion and submit dependent nodes

    Attributes
    ----------
    nodes : dict[str, WorkflowNode]
        All nodes in the plan, keyed by their unique key.
    root_key : str
        Key of the root node (the one the user requested).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    nodes: dict[str, WorkflowNode]
    root_key: str

    def ready_nodes(self, completed: set[str]) -> list[WorkflowNode]:
        """Get nodes whose dependencies are all satisfied.

        Parameters
        ----------
        completed : set[str]
            Keys of nodes that have completed successfully.

        Returns
        -------
        list[WorkflowNode]
            Nodes ready to execute.
        """
        ready = []
        for node in self.nodes.values():
            # Skip if already completed or cached
            if node.key in completed or node.cached:
                continue
            # Check all dependencies are satisfied
            all_deps_ready = all(
                dep in completed or self.nodes[dep].cached
                for dep in node.depends_on
            )
            if all_deps_ready:
                ready.append(node)
        return ready

    def leaf_nodes(self) -> list[WorkflowNode]:
        """Get nodes with no dependencies (starting points).

        Returns
        -------
        list[WorkflowNode]
            Nodes that can start immediately.
        """
        return [
            node for node in self.nodes.values()
            if not node.depends_on and not node.cached
        ]

    def is_complete(self, completed: set[str]) -> bool:
        """Check if the entire plan has completed.

        Parameters
        ----------
        completed : set[str]
            Keys of nodes that have completed successfully.

        Returns
        -------
        bool
            True if root node is complete or cached.
        """
        root = self.nodes[self.root_key]
        return root.key in completed or root.cached

    def get_dependency_prefixes(self, node_key: str) -> dict[str, str]:
        """Get storage prefixes for a node's dependencies.

        Parameters
        ----------
        node_key : str
            Key of the node.

        Returns
        -------
        dict[str, str]
            Mapping from dependency key to storage prefix.
        """
        node = self.nodes[node_key]
        return {
            dep_key: self.nodes[dep_key].storage_prefix
            for dep_key in node.depends_on
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowPlan:
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, s: str) -> WorkflowPlan:
        """Deserialize from JSON string."""
        return cls.model_validate_json(s)


def compute_storage_prefix(
    function_name: str,
    subject_key: str,
    kwargs: dict,
    base_prefix: str = "muflow",
) -> str:
    """Compute deterministic, content-addressed storage prefix.

    .. deprecated::
        Use ``muflow.storage.compute_prefix`` instead.
    """
    import warnings

    warnings.warn(
        "compute_storage_prefix is deprecated. "
        "Use muflow.storage.compute_prefix instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from muflow.storage.base import compute_prefix

    hash_dict = {
        "workflow": function_name,
        "subject": subject_key,
        **kwargs,
    }
    return compute_prefix(hash_dict, base_prefix=base_prefix)


def compute_node_key(
    function_name: str,
    subject_key: str,
    kwargs: dict,
) -> str:
    """Compute a unique key for a workflow node.

    .. deprecated::
        Use ``muflow.storage.compute_prefix`` instead.
    """
    import warnings

    warnings.warn(
        "compute_node_key is deprecated. "
        "Use muflow.storage.compute_prefix instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from muflow.storage.base import compute_prefix

    hash_dict = {
        "workflow": function_name,
        "subject": subject_key,
        **kwargs,
    }
    prefix = compute_prefix(hash_dict)
    # Return just the hash portion as a short key
    return f"{function_name}/{subject_key}/{prefix.rsplit('/', 1)[-1][:8]}"
