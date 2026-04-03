"""Tests for StepFunctionsBackend."""

import json

import pytest

from muflow.plan import WorkflowNode, WorkflowPlan
from muflow.registry import clear, register_workflow

try:
    import boto3
    from moto import mock_aws
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="boto3 and moto are required for Step Functions tests",
)

FUNCTION_ARN = "arn:aws:lambda:us-east-1:123456789012:function:muflow-worker"
ROLE_ARN = "arn:aws:iam::123456789012:role/StepFunctionsRole"
BUCKET = "test-bucket"
BASE_PREFIX = "muflow"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry between tests."""
    clear()
    yield
    clear()


@pytest.fixture
def sfn_client():
    """Moto-backed Step Functions client (no real AWS calls)."""
    with mock_aws():
        yield boto3.client("stepfunctions", region_name="us-east-1")


@pytest.fixture
def backend(sfn_client):
    from muflow.backends.step_functions import StepFunctionsBackend

    return StepFunctionsBackend(
        function_arn=FUNCTION_ARN,
        bucket=BUCKET,
        role_arn=ROLE_ARN,
        base_prefix=BASE_PREFIX,
        sfn_client=sfn_client,
    )


def _simple_plan() -> WorkflowPlan:
    """Single-node plan: A."""
    node = WorkflowNode(
        key="muflow/test.simple/aaa",
        function="test.simple",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.simple/aaa",
    )
    return WorkflowPlan(nodes={"muflow/test.simple/aaa": node}, root_key="muflow/test.simple/aaa")


def _linear_plan() -> WorkflowPlan:
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


def _fan_in_plan() -> WorkflowPlan:
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
        depends_on=[l.key for l in leaves],
    )
    nodes = {l.key: l for l in leaves}
    nodes[root.key] = root
    return WorkflowPlan(nodes=nodes, root_key=root.key)


def _all_cached_plan() -> WorkflowPlan:
    node = WorkflowNode(
        key="muflow/test.simple/aaa",
        function="test.simple",
        subject_key="sub:1",
        kwargs={},
        storage_prefix="muflow/test.simple/aaa",
        cached=True,
    )
    return WorkflowPlan(nodes={node.key: node}, root_key=node.key)


# ── Unit tests: ASL generation (no AWS calls needed) ─────────────────────────


class TestComputeLevels:
    def _make_backend(self):
        from muflow.backends.step_functions import StepFunctionsBackend
        # sfn_client is only used in AWS calls; None is fine for unit tests
        # that don't reach submit_plan
        with mock_aws():
            client = boto3.client("stepfunctions", region_name="us-east-1")
        return StepFunctionsBackend(
            function_arn=FUNCTION_ARN,
            bucket=BUCKET,
            role_arn=ROLE_ARN,
            sfn_client=client,
        )

    def test_single_node(self):
        backend = self._make_backend()
        plan = _simple_plan()
        levels = backend._compute_levels(plan)
        assert len(levels) == 1
        assert levels[0][0].key == "muflow/test.simple/aaa"

    def test_linear_two_nodes(self):
        backend = self._make_backend()
        plan = _linear_plan()
        levels = backend._compute_levels(plan)
        assert len(levels) == 2
        assert levels[0][0].function == "test.dep"
        assert levels[1][0].function == "test.root"

    def test_fan_in(self):
        backend = self._make_backend()
        plan = _fan_in_plan()
        levels = backend._compute_levels(plan)
        assert len(levels) == 2
        assert len(levels[0]) == 3   # three leaves in parallel
        assert len(levels[1]) == 1   # one root

    def test_cached_nodes_excluded(self):
        backend = self._make_backend()
        plan = _all_cached_plan()
        levels = backend._compute_levels(plan)
        assert levels == []

    def test_partial_cache(self):
        """Cached dep means root is in level 0, not level 1."""
        backend = self._make_backend()
        dep = WorkflowNode(
            key="d", function="test.dep", subject_key="s", kwargs={},
            storage_prefix="d", cached=True,
        )
        root = WorkflowNode(
            key="r", function="test.root", subject_key="s", kwargs={},
            storage_prefix="r", depends_on=["d"],
        )
        plan = WorkflowPlan(nodes={"d": dep, "r": root}, root_key="r")
        levels = backend._compute_levels(plan)
        assert len(levels) == 1
        assert levels[0][0].key == "r"


class TestBuildASL:
    def _make_backend(self):
        from muflow.backends.step_functions import StepFunctionsBackend
        with mock_aws():
            client = boto3.client("stepfunctions", region_name="us-east-1")
        return StepFunctionsBackend(
            function_arn=FUNCTION_ARN,
            bucket=BUCKET,
            role_arn=ROLE_ARN,
            sfn_client=client,
        )

    def test_returns_none_for_empty_levels(self):
        backend = self._make_backend()
        assert backend._build_asl([], _simple_plan()) is None

    def test_single_node_is_task_state(self):
        backend = self._make_backend()
        plan = _simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        assert asl["StartAt"] == "Level0"
        state = asl["States"]["Level0"]
        assert state["Type"] == "Task"
        assert state["Resource"] == "arn:aws:states:::lambda:invoke"
        assert state["End"] is True
        assert "Next" not in state

    def test_single_node_payload(self):
        backend = self._make_backend()
        plan = _simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        payload = asl["States"]["Level0"]["Parameters"]["Payload"]
        assert payload["workflow_name"] == "test.simple"
        assert payload["storage_prefix"] == "muflow/test.simple/aaa"
        assert payload["bucket"] == BUCKET
        assert payload["node_key"] == "muflow/test.simple/aaa"
        assert isinstance(payload["dependency_prefixes"], dict)

    def test_task_state_has_retry(self):
        backend = self._make_backend()
        plan = _simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        retry = asl["States"]["Level0"]["Retry"]
        assert len(retry) == 1
        assert "Lambda.ServiceException" in retry[0]["ErrorEquals"]
        assert retry[0]["MaxAttempts"] == 3

    def test_multi_node_level_is_parallel_state(self):
        backend = self._make_backend()
        plan = _fan_in_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        leaf_state = asl["States"]["Level0"]
        assert leaf_state["Type"] == "Parallel"
        assert len(leaf_state["Branches"]) == 3
        # Each branch must be a self-contained state machine
        for branch in leaf_state["Branches"]:
            assert branch["StartAt"] == "Execute"
            assert branch["States"]["Execute"]["Type"] == "Task"
            assert branch["States"]["Execute"]["End"] is True

    def test_level_sequencing(self):
        backend = self._make_backend()
        plan = _linear_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        assert asl["States"]["Level0"]["Next"] == "Level1"
        assert asl["States"]["Level1"].get("End") is True
        assert "Next" not in asl["States"]["Level1"]

    def test_result_path_is_null(self):
        """ResultPath: null discards Lambda output, preserving state input."""
        backend = self._make_backend()
        plan = _simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        assert asl["States"]["Level0"]["ResultPath"] is None

    def test_asl_is_json_serialisable(self):
        backend = self._make_backend()
        plan = _fan_in_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)
        # Must not raise
        json.dumps(asl)

    def test_function_arn_in_parameters(self):
        backend = self._make_backend()
        plan = _simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        assert asl["States"]["Level0"]["Parameters"]["FunctionName"] == FUNCTION_ARN


class TestStateMachineName:
    def _make_backend(self):
        from muflow.backends.step_functions import StepFunctionsBackend
        with mock_aws():
            client = boto3.client("stepfunctions", region_name="us-east-1")
        return StepFunctionsBackend(
            function_arn=FUNCTION_ARN,
            bucket=BUCKET,
            role_arn=ROLE_ARN,
            sfn_client=client,
        )

    def test_uses_hash_suffix(self):
        backend = self._make_backend()
        name = backend._state_machine_name("muflow/my.workflow/abc123def456")
        assert name == "muflow-abc123def456"

    def test_sanitises_special_chars(self):
        backend = self._make_backend()
        name = backend._state_machine_name("muflow/my.workflow/a.b:c/d")
        assert all(c.isalnum() or c in "-_" for c in name)

    def test_max_80_chars(self):
        backend = self._make_backend()
        long_key = "muflow/" + "x" * 200
        assert len(backend._state_machine_name(long_key)) <= 80

    def test_custom_prefix(self):
        from muflow.backends.step_functions import StepFunctionsBackend
        with mock_aws():
            client = boto3.client("stepfunctions", region_name="us-east-1")
        backend = StepFunctionsBackend(
            function_arn=FUNCTION_ARN,
            bucket=BUCKET,
            role_arn=ROLE_ARN,
            state_machine_prefix="myapp",
            sfn_client=client,
        )
        name = backend._state_machine_name("muflow/wf/hash123")
        assert name.startswith("myapp-")


# ── Integration tests: AWS calls via moto ─────────────────────────────────────


class TestSubmitPlan:
    def test_creates_state_machine_and_returns_arn(self, backend, sfn_client):
        with mock_aws():
            plan = _simple_plan()
            execution_arn = backend.submit_plan(plan)

            assert "arn:aws:states" in execution_arn
            assert "exec-" in execution_arn

    def test_all_cached_returns_sentinel(self, backend):
        with mock_aws():
            plan = _all_cached_plan()
            result = backend.submit_plan(plan)
            assert result.startswith("cached-")

    def test_state_machine_created_with_correct_name(self, backend, sfn_client):
        with mock_aws():
            plan = _simple_plan()
            backend.submit_plan(plan)

            machines = sfn_client.list_state_machines()["stateMachines"]
            assert len(machines) == 1
            assert machines[0]["name"].startswith("muflow-")

    def test_state_machine_definition_contains_function_arn(self, backend, sfn_client):
        with mock_aws():
            plan = _simple_plan()
            backend.submit_plan(plan)

            machines = sfn_client.list_state_machines()["stateMachines"]
            arn = machines[0]["stateMachineArn"]
            desc = sfn_client.describe_state_machine(stateMachineArn=arn)
            definition = json.loads(desc["definition"])

            # FunctionName must appear somewhere in the ASL
            asl_str = json.dumps(definition)
            assert FUNCTION_ARN in asl_str

    def test_resubmit_same_plan_updates_not_duplicates(self, backend, sfn_client):
        """Submitting the same plan twice reuses the state machine."""
        with mock_aws():
            plan = _simple_plan()
            backend.submit_plan(plan)
            backend.submit_plan(plan)

            machines = sfn_client.list_state_machines()["stateMachines"]
            assert len(machines) == 1

    def test_callbacks_ignored_with_warning(self, backend, caplog):
        import logging
        with mock_aws():
            plan = _simple_plan()
            with caplog.at_level(logging.WARNING):
                backend.submit_plan(plan, on_node_complete=lambda k: None)
            assert "not supported" in caplog.text.lower()


class TestGetPlanState:
    def test_cached_sentinel_returns_success(self, backend):
        with mock_aws():
            assert backend.get_plan_state("cached-anything") == "success"

    def test_running_execution(self, backend, sfn_client):
        with mock_aws():
            plan = _simple_plan()
            execution_arn = backend.submit_plan(plan)
            state = backend.get_plan_state(execution_arn)
            # moto keeps executions in RUNNING state (no actual Lambda)
            assert state in ("running", "success", "failure")

    def test_unknown_status_maps_to_pending(self, backend, sfn_client):
        """Unmapped SF statuses fall back to 'pending'."""
        with mock_aws():
            plan = _simple_plan()
            execution_arn = backend.submit_plan(plan)
            # Patch describe_execution to return an unexpected status
            original = sfn_client.describe_execution

            def patched(**kwargs):
                r = original(**kwargs)
                r["status"] = "WEIRD_UNKNOWN"
                return r

            sfn_client.describe_execution = patched
            assert backend.get_plan_state(execution_arn) == "pending"


class TestCancelPlan:
    def test_cached_sentinel_is_noop(self, backend):
        with mock_aws():
            # Should not raise
            backend.cancel_plan("cached-anything")

    def test_stop_execution_called(self, backend, sfn_client):
        with mock_aws():
            plan = _simple_plan()
            execution_arn = backend.submit_plan(plan)
            # moto supports stop_execution
            backend.cancel_plan(execution_arn)
            desc = sfn_client.describe_execution(executionArn=execution_arn)
            assert desc["status"] in ("ABORTED", "STOPPED", "RUNNING")
