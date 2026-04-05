"""Task plan data structures.

A TaskPlan represents a complete, static execution DAG. It is computed
once upfront and stored as JSON. The plan contains all information needed
to execute the task on any backend.
"""

from __future__ import annotations

from typing import Optional

import pydantic


class TaskNode(pydantic.BaseModel):
    """A single node in the execution plan.

    Attributes
    ----------
    key : str
        Unique identifier within the plan. Typically derived from
        function name, subject, and kwargs.
    function : str
        Task function name (e.g., "sds_ml.v3.gpr.training").
    subject_key : str
        Identifier for the subject (e.g., S3 key or "tag:123").
    kwargs : dict
        Parameters passed to the task.
    storage_prefix : str
        Content-addressed S3 key prefix for output files.
    depends_on : list[str]
        Keys of upstream nodes that must complete first.
    depended_on_by : list[str]
        Keys of downstream nodes waiting on this one.
    output_files : list[str]
        Filenames this node will produce (from Outputs schema).
    analysis_id : int, optional
        Database ID of the TaskResult (set after DB record creation).
    dependency_access_map : dict[str, str]
        Mapping from access key (e.g., ``"features:0"``) to storage prefix.
        Populated at plan-build time by the Pipeline or TaskPlanner.
        Used by backends to set up ``TaskContext`` with dependency storages.
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
    analysis_id: Optional[int] = None
    dependency_access_map: dict[str, str] = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> TaskNode:
        """Create from dictionary."""
        return cls.model_validate(data)


class TaskPlan(pydantic.BaseModel):
    """A complete, static execution DAG.

    The plan is computed once by the TaskPlanner and stored as JSON.
    It contains all information needed to:
    - Create write-ahead database records
    - Submit nodes to execution backends
    - Track completion and submit dependent nodes

    Attributes
    ----------
    nodes : dict[str, TaskNode]
        All nodes in the plan, keyed by their unique key.
    root_key : str
        Key of the root node (the one the user requested).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    nodes: dict[str, TaskNode]
    root_key: str

    def ready_nodes(self, completed: set[str]) -> list[TaskNode]:
        """Get nodes whose dependencies are all satisfied.

        Parameters
        ----------
        completed : set[str]
            Keys of nodes that have completed successfully.

        Returns
        -------
        list[TaskNode]
            Nodes ready to execute.
        """
        ready = []
        for node in self.nodes.values():
            if node.key in completed:
                continue
            if all(dep in completed for dep in node.depends_on):
                ready.append(node)
        return ready

    def leaf_nodes(self) -> list[TaskNode]:
        """Get nodes with no dependencies (starting points).

        Returns
        -------
        list[TaskNode]
            Nodes that can start immediately.
        """
        return [node for node in self.nodes.values() if not node.depends_on]

    def is_complete(self, completed: set[str]) -> bool:
        """Check if the entire plan has completed.

        Parameters
        ----------
        completed : set[str]
            Keys of nodes that have completed successfully.

        Returns
        -------
        bool
            True if root node is complete.
        """
        return self.root_key in completed

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: dict) -> TaskPlan:
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, s: str) -> TaskPlan:
        """Deserialize from JSON string."""
        return cls.model_validate_json(s)
