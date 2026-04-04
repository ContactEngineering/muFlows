"""Shared test fixtures and plan factories."""

import pytest

from muflow.plan import WorkflowNode, WorkflowPlan
from muflow.registry import clear


@pytest.fixture(autouse=False)
def clean_registry():
    """Reset workflow registry between tests."""
    clear()
    yield
    clear()


def simple_plan() -> WorkflowPlan:
    """Single-node plan: A."""
    node = WorkflowNode(
        key="muflow/test.simple/aaa",
        function="test.simple",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.simple/aaa",
    )
    return WorkflowPlan(nodes={node.key: node}, root_key=node.key)


def linear_plan() -> WorkflowPlan:
    """Two-node linear plan: dep → root."""
    dep = WorkflowNode(
        key="muflow/test.dep/bbb",
        function="test.dep",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.dep/bbb",
    )
    root = WorkflowNode(
        key="muflow/test.root/ccc",
        function="test.root",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.root/ccc",
        depends_on=["muflow/test.dep/bbb"],
    )
    return WorkflowPlan(
        nodes={dep.key: dep, root.key: root},
        root_key=root.key,
    )


def fan_in_plan() -> WorkflowPlan:
    """Three leaf nodes feeding one root: leaf0, leaf1, leaf2 → root."""
    leaves = [
        WorkflowNode(
            key=f"muflow/test.leaf/l{i}",
            function="test.leaf",
            subject_key=f"sub:{i}",
            kwargs={"i": i},
            storage_prefix=f"muflow/test.leaf/l{i}",
        )
        for i in range(3)
    ]
    root = WorkflowNode(
        key="muflow/test.root/rrr",
        function="test.root",
        subject_key="sub:all",
        kwargs={},
        storage_prefix="muflow/test.root/rrr",
        depends_on=[node.key for node in leaves],
    )
    nodes = {node.key: node for node in leaves}
    nodes[root.key] = root
    return WorkflowPlan(nodes=nodes, root_key=root.key)


def diamond_plan() -> WorkflowPlan:
    """Diamond DAG: A → B, C → D (4 nodes, 3 levels)."""
    a = WorkflowNode(
        key="muflow/test.a/aaa",
        function="test.a",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.a/aaa",
    )
    b = WorkflowNode(
        key="muflow/test.b/bbb",
        function="test.b",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.b/bbb",
        depends_on=["muflow/test.a/aaa"],
    )
    c = WorkflowNode(
        key="muflow/test.c/ccc",
        function="test.c",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.c/ccc",
        depends_on=["muflow/test.a/aaa"],
    )
    d = WorkflowNode(
        key="muflow/test.d/ddd",
        function="test.d",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.d/ddd",
        depends_on=["muflow/test.b/bbb", "muflow/test.c/ccc"],
    )
    nodes = {n.key: n for n in [a, b, c, d]}
    return WorkflowPlan(nodes=nodes, root_key=d.key)


def all_cached_plan() -> WorkflowPlan:
    """Single cached node."""
    node = WorkflowNode(
        key="muflow/test.simple/aaa",
        function="test.simple",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.simple/aaa",
        cached=True,
    )
    return WorkflowPlan(nodes={node.key: node}, root_key=node.key)


def partial_cache_plan() -> WorkflowPlan:
    """Cached dep → non-cached root."""
    dep = WorkflowNode(
        key="d",
        function="test.dep",
        subject_key="s",
        kwargs={},
        storage_prefix="d",
        cached=True,
    )
    root = WorkflowNode(
        key="r",
        function="test.root",
        subject_key="s",
        kwargs={},
        storage_prefix="r",
        depends_on=["d"],
    )
    return WorkflowPlan(nodes={"d": dep, "r": root}, root_key="r")
