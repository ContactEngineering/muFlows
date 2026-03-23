"""Tests for the pure execution function."""

import tempfile
from pathlib import Path

import pytest

from muflows import LocalFolderContext, WorkflowImplementation
from muflows.executor import ExecutionPayload, ExecutionResult, execute_workflow


class MockWorkflow(WorkflowImplementation):
    """A simple test workflow that writes a result file."""

    class Meta:
        name = "test.mock_workflow"

    class Parameters(WorkflowImplementation.Parameters):
        user_id: int = 0

    def execute(self, ctx):
        ctx.save_json("result.json", {"status": "ok", "user_id": self.kwargs.user_id})


class FailingWorkflow(WorkflowImplementation):
    """A workflow that always fails."""

    class Meta:
        name = "test.failing_workflow"

    def execute(self, ctx):
        raise ValueError("Intentional failure for testing")


# Simple registry for tests
TEST_REGISTRY = {
    "test.mock_workflow": MockWorkflow,
    "test.failing_workflow": FailingWorkflow,
}


def get_test_implementation(name):
    """Test implementation getter."""
    return TEST_REGISTRY[name]


class TestExecutionPayload:
    """Tests for ExecutionPayload dataclass."""

    def test_creation(self):
        """Should create payload with required fields."""
        payload = ExecutionPayload(
            workflow_name="test.workflow",
            kwargs={"param": "value"},
            storage_prefix="results/test",
        )
        assert payload.workflow_name == "test.workflow"
        assert payload.kwargs == {"param": "value"}
        assert payload.storage_prefix == "results/test"
        assert payload.dependency_prefixes == {}
        assert payload.allowed_outputs == set()

    def test_to_dict(self):
        """Should serialize to dictionary."""
        payload = ExecutionPayload(
            workflow_name="test.workflow",
            kwargs={"param": "value"},
            storage_prefix="results/test",
            dependency_prefixes={"dep1": "results/dep1"},
            allowed_outputs={"result.json", "model.nc"},
        )
        d = payload.to_dict()

        assert d["workflow_name"] == "test.workflow"
        assert d["kwargs"] == {"param": "value"}
        assert d["storage_prefix"] == "results/test"
        assert d["dependency_prefixes"] == {"dep1": "results/dep1"}
        assert set(d["allowed_outputs"]) == {"result.json", "model.nc"}

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "workflow_name": "test.workflow",
            "kwargs": {"param": "value"},
            "storage_prefix": "results/test",
            "dependency_prefixes": {"dep1": "results/dep1"},
            "allowed_outputs": ["result.json", "model.nc"],
        }
        payload = ExecutionPayload.from_dict(d)

        assert payload.workflow_name == "test.workflow"
        assert payload.kwargs == {"param": "value"}
        assert payload.storage_prefix == "results/test"
        assert payload.dependency_prefixes == {"dep1": "results/dep1"}
        assert payload.allowed_outputs == {"result.json", "model.nc"}

    def test_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = ExecutionPayload(
            workflow_name="test.workflow",
            kwargs={"param": "value"},
            storage_prefix="results/test",
            dependency_prefixes={"dep1": "results/dep1"},
            allowed_outputs={"result.json"},
        )
        restored = ExecutionPayload.from_dict(original.to_dict())

        assert restored.workflow_name == original.workflow_name
        assert restored.kwargs == original.kwargs
        assert restored.storage_prefix == original.storage_prefix
        assert restored.dependency_prefixes == original.dependency_prefixes
        assert restored.allowed_outputs == original.allowed_outputs


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        """Should create success result."""
        result = ExecutionResult(
            success=True,
            files_written=["result.json"],
        )
        assert result.success is True
        assert result.error_message is None
        assert result.error_traceback is None
        assert result.files_written == ["result.json"]

    def test_failure_result(self):
        """Should create failure result."""
        result = ExecutionResult(
            success=False,
            error_message="Something went wrong",
            error_traceback="Traceback...",
        )
        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.error_traceback == "Traceback..."

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = ExecutionResult(
            success=True,
            files_written=["result.json"],
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["error_message"] is None
        assert d["files_written"] == ["result.json"]

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            "success": False,
            "error_message": "Error",
            "error_traceback": "Traceback",
            "files_written": [],
        }
        result = ExecutionResult.from_dict(d)

        assert result.success is False
        assert result.error_message == "Error"
        assert result.error_traceback == "Traceback"


class TestExecuteWorkflow:
    """Tests for execute_workflow function."""

    def test_successful_execution(self):
        """Should execute workflow and return success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.mock_workflow",
                kwargs={"user_id": 42},
                storage_prefix=tmpdir,
                allowed_outputs={"result.json"},
            )

            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs=payload.kwargs,
                allowed_outputs=payload.allowed_outputs,
            )

            result = execute_workflow(payload, ctx, get_test_implementation)

            assert result.success is True
            assert result.error_message is None
            assert "result.json" in result.files_written

            # Verify file was written
            data = ctx.read_json("result.json")
            assert data["status"] == "ok"
            assert data["user_id"] == 42

    def test_failed_execution(self):
        """Should catch exceptions and return failure result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.failing_workflow",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs={},
            )

            result = execute_workflow(payload, ctx, get_test_implementation)

            assert result.success is False
            assert "Intentional failure" in result.error_message
            assert result.error_traceback is not None
            assert "ValueError" in result.error_traceback

    def test_unknown_workflow(self):
        """Should return failure for unknown workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="nonexistent.workflow",
                kwargs={},
                storage_prefix=tmpdir,
            )

            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs={},
            )

            result = execute_workflow(payload, ctx, get_test_implementation)

            assert result.success is False
            assert "nonexistent.workflow" in result.error_message

    def test_output_guard_violation(self):
        """Should fail if workflow writes undeclared file."""

        class BadWorkflow(WorkflowImplementation):
            class Meta:
                name = "test.bad_workflow"

            def execute(self, ctx):
                ctx.save_json("undeclared.json", {})

        registry = {"test.bad_workflow": BadWorkflow}

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.bad_workflow",
                kwargs={},
                storage_prefix=tmpdir,
                allowed_outputs={"allowed.json"},  # Not 'undeclared.json'
            )

            ctx = LocalFolderContext(
                path=tmpdir,
                kwargs={},
                allowed_outputs=payload.allowed_outputs,
            )

            result = execute_workflow(payload, ctx, lambda name: registry[name])

            assert result.success is False
            assert "undeclared.json" in result.error_message
            assert "PermissionError" in result.error_traceback
