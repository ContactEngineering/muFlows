"""Shared test fixtures and plan factories."""

import pytest

from muflow.plan import TaskNode, TaskPlan
from muflow.registry import clear


@pytest.fixture(autouse=False)
def clean_registry():
    """Reset task registry between tests."""
    clear()
    yield
    clear()


def simple_plan() -> TaskPlan:
    """Single-node plan: A."""
    node = TaskNode(
        key="muflow/test.simple/aaa",
        function="test.simple",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.simple/aaa",
    )
    return TaskPlan(nodes={node.key: node}, root_key=node.key)


def linear_plan() -> TaskPlan:
    """Two-node linear plan: dep → root."""
    dep = TaskNode(
        key="muflow/test.dep/bbb",
        function="test.dep",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.dep/bbb",
    )
    root = TaskNode(
        key="muflow/test.root/ccc",
        function="test.root",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.root/ccc",
        depends_on=["muflow/test.dep/bbb"],
    )
    return TaskPlan(
        nodes={dep.key: dep, root.key: root},
        root_key=root.key,
    )


def fan_in_plan() -> TaskPlan:
    """Three leaf nodes feeding one root: leaf0, leaf1, leaf2 → root."""
    leaves = [
        TaskNode(
            key=f"muflow/test.leaf/l{i}",
            function="test.leaf",
            subject_key=f"sub:{i}",
            kwargs={"i": i},
            storage_prefix=f"muflow/test.leaf/l{i}",
        )
        for i in range(3)
    ]
    root = TaskNode(
        key="muflow/test.root/rrr",
        function="test.root",
        subject_key="sub:all",
        kwargs={},
        storage_prefix="muflow/test.root/rrr",
        depends_on=[node.key for node in leaves],
    )
    nodes = {node.key: node for node in leaves}
    nodes[root.key] = root
    return TaskPlan(nodes=nodes, root_key=root.key)


def diamond_plan() -> TaskPlan:
    """Diamond DAG: A → B, C → D (4 nodes, 3 levels)."""
    a = TaskNode(
        key="muflow/test.a/aaa",
        function="test.a",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.a/aaa",
    )
    b = TaskNode(
        key="muflow/test.b/bbb",
        function="test.b",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.b/bbb",
        depends_on=["muflow/test.a/aaa"],
    )
    c = TaskNode(
        key="muflow/test.c/ccc",
        function="test.c",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.c/ccc",
        depends_on=["muflow/test.a/aaa"],
    )
    d = TaskNode(
        key="muflow/test.d/ddd",
        function="test.d",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.d/ddd",
        depends_on=["muflow/test.b/bbb", "muflow/test.c/ccc"],
    )
    nodes = {n.key: n for n in [a, b, c, d]}
    return TaskPlan(nodes=nodes, root_key=d.key)


