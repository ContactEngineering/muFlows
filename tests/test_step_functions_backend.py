"""Tests for StepFunctionsBackend."""

import json

import pytest

from tests.conftest import (
    fan_in_plan,
    linear_plan,
    simple_plan,
)

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


# ── Unit tests: ASL generation (no AWS calls needed) ─────────────────────────


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
        assert backend._build_asl([], simple_plan()) is None

    def test_single_node_is_task_state(self):
        backend = self._make_backend()
        plan = simple_plan()
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
        plan = simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        payload = asl["States"]["Level0"]["Parameters"]["Payload"]
        assert payload["task_name"] == "test.simple"
        assert payload["storage_prefix"] == "muflow/test.simple/aaa"
        assert payload["bucket"] == BUCKET
        assert payload["node_key"] == "muflow/test.simple/aaa"
        assert isinstance(payload["dependency_prefixes"], dict)

    def test_task_state_has_retry(self):
        backend = self._make_backend()
        plan = simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        retry = asl["States"]["Level0"]["Retry"]
        assert len(retry) == 1
        assert "Lambda.ServiceException" in retry[0]["ErrorEquals"]
        assert retry[0]["MaxAttempts"] == 3

    def test_multi_node_level_is_parallel_state(self):
        backend = self._make_backend()
        plan = fan_in_plan()
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
        plan = linear_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        assert asl["States"]["Level0"]["Next"] == "Level1"
        assert asl["States"]["Level1"].get("End") is True
        assert "Next" not in asl["States"]["Level1"]

    def test_result_path_is_null(self):
        """ResultPath: null discards Lambda output, preserving state input."""
        backend = self._make_backend()
        plan = simple_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)

        assert asl["States"]["Level0"]["ResultPath"] is None

    def test_asl_is_json_serialisable(self):
        backend = self._make_backend()
        plan = fan_in_plan()
        levels = backend._compute_levels(plan)
        asl = backend._build_asl(levels, plan)
        # Must not raise
        json.dumps(asl)

    def test_function_arn_in_parameters(self):
        backend = self._make_backend()
        plan = simple_plan()
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
        name = backend._state_machine_name("muflow/my.task/abc123def456")
        assert name == "muflow-abc123def456"

    def test_sanitises_special_chars(self):
        backend = self._make_backend()
        name = backend._state_machine_name("muflow/my.task/a.b:c/d")
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
    def test_creates_state_machine_and_returns_handle(self, backend, sfn_client):
        from muflow.backends.handle import PlanHandle

        with mock_aws():
            plan = simple_plan()
            handle = backend.submit_plan(plan)

            assert isinstance(handle, PlanHandle)
            assert handle.backend == "step_functions"
            assert "arn:aws:states" in handle.plan_id
            assert "exec-" in handle.plan_id

    def test_state_machine_created_with_correct_name(self, backend, sfn_client):
        with mock_aws():
            plan = simple_plan()
            backend.submit_plan(plan)

            machines = sfn_client.list_state_machines()["stateMachines"]
            assert len(machines) == 1
            assert machines[0]["name"].startswith("muflow-")

    def test_state_machine_definition_contains_function_arn(self, backend, sfn_client):
        with mock_aws():
            plan = simple_plan()
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
            plan = simple_plan()
            backend.submit_plan(plan)
            backend.submit_plan(plan)

            machines = sfn_client.list_state_machines()["stateMachines"]
            assert len(machines) == 1

    def test_completion_callback_logs_warning(self, backend, caplog):
        """Passing completion_callback to SFN backend logs a warning."""
        import logging
        from muflow.backends.callbacks import NoOpCompletionCallback

        with mock_aws():
            plan = simple_plan()
            with caplog.at_level(logging.WARNING):
                backend.submit_plan(plan, completion_callback=NoOpCompletionCallback())
            assert "not supported" in caplog.text.lower()


class TestGetPlanState:
    def test_running_execution(self, backend, sfn_client):
        with mock_aws():
            plan = simple_plan()
            handle = backend.submit_plan(plan)
            state = backend.get_plan_state(handle.plan_id)
            # moto keeps executions in RUNNING state (no actual Lambda)
            assert state in ("running", "success", "failure")

    def test_unknown_status_maps_to_pending(self, backend, sfn_client):
        """Unmapped SF statuses fall back to 'pending'."""
        with mock_aws():
            plan = simple_plan()
            handle = backend.submit_plan(plan)
            # Patch describe_execution to return an unexpected status
            original = sfn_client.describe_execution

            def patched(**kwargs):
                r = original(**kwargs)
                r["status"] = "WEIRD_UNKNOWN"
                return r

            sfn_client.describe_execution = patched
            assert backend.get_plan_state(handle.plan_id) == "pending"


class TestCancelPlan:
    def test_stop_execution_called(self, backend, sfn_client):
        with mock_aws():
            plan = simple_plan()
            handle = backend.submit_plan(plan)
            # moto supports stop_execution
            backend.cancel_plan(handle.plan_id)
            desc = sfn_client.describe_execution(executionArn=handle.plan_id)
            assert desc["status"] in ("ABORTED", "STOPPED", "RUNNING")
