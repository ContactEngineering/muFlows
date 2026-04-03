"""Tests for function-based workflow registration and execution."""

import tempfile

import pydantic
import pytest

from muflow import create_local_context
from muflow.executor import ExecutionPayload, execute_workflow
from muflow.registry import (
    AlreadyRegisteredException,
    WorkflowEntry,
    clear,
    get,
    get_all,
    get_names,
    register_workflow,
    unregister,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the registry before and after each test."""
    clear()
    yield
    clear()


# ── Registration tests ──────────────────────────────────────────────────────


class TestRegisterWorkflow:
    def test_decorator_registers_function(self):
        @register_workflow(name="test.simple")
        def simple(context):
            pass

        entry = get("test.simple")
        assert entry is not None
        assert entry.name == "test.simple"
        assert entry.fn is simple

    def test_decorator_with_all_metadata(self):
        from typing import Annotated
        from muflow import IdentityKey

        class Params(pydantic.BaseModel):
            threshold: Annotated[float, IdentityKey()] = 0.5

        @register_workflow(
            name="test.full",
            display_name="Full Test",
            queue="analysis",
            parameters=Params,
        )
        def full(context):
            pass

        entry = get("test.full")
        assert entry.display_name == "Full Test"
        assert entry.queue == "analysis"
        assert entry.parameters is Params
        assert entry.identity_keys == ["threshold"]

    def test_decorator_returns_original_function(self):
        @register_workflow(name="test.identity")
        def my_fn(context):
            return "hello"

        # The decorator should return the original function
        assert my_fn(None) == "hello"

    def test_duplicate_name_raises(self):
        @register_workflow(name="test.dup")
        def first(context):
            pass

        with pytest.raises(AlreadyRegisteredException):

            @register_workflow(name="test.dup")
            def second(context):
                pass

    def test_get_all_returns_entries(self):
        @register_workflow(name="test.a")
        def a(context):
            pass

        @register_workflow(name="test.b")
        def b(context):
            pass

        all_entries = get_all()
        assert len(all_entries) == 2
        assert isinstance(all_entries["test.a"], WorkflowEntry)
        assert isinstance(all_entries["test.b"], WorkflowEntry)

    def test_get_names(self):
        @register_workflow(name="test.x")
        def x(context):
            pass

        assert "test.x" in get_names()

    def test_unregister(self):
        @register_workflow(name="test.remove")
        def rm(context):
            pass

        assert get("test.remove") is not None
        unregister("test.remove")
        assert get("test.remove") is None


# ── Execution tests ─────────────────────────────────────────────────────────


class TestFunctionWorkflowExecution:
    def test_simple_execution(self):
        @register_workflow(name="test.greet")
        def greet(context):
            context.save_json("result.json", {"hello": "world"})

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.greet",
                kwargs={},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_workflow(payload, ctx, lambda name: get(name))

            assert result.success is True
            assert "result.json" in result.files_written
            assert ctx.read_json("result.json") == {"hello": "world"}

    def test_execution_with_parameters(self):
        class MyParams(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(extra="forbid")
            scale: float = 1.0

        @register_workflow(
            name="test.scaled",
            parameters=MyParams,
        )
        def scaled(context):
            s = context.kwargs.scale
            context.save_json("result.json", {"scaled_value": 10 * s})

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.scaled",
                kwargs={"scale": 2.5},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            result = execute_workflow(payload, ctx, lambda name: get(name))

            assert result.success is True
            data = ctx.read_json("result.json")
            assert data["scaled_value"] == 25.0

    def test_parameter_validation_failure(self):
        class StrictParams(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(extra="forbid")
            required_field: int

        @register_workflow(
            name="test.strict",
            parameters=StrictParams,
        )
        def strict(context):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.strict",
                kwargs={},  # Missing required_field
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs=payload.kwargs)

            result = execute_workflow(payload, ctx, lambda name: get(name))

            assert result.success is False
            assert "required_field" in result.error_message

    def test_execution_failure(self):
        @register_workflow(name="test.boom")
        def boom(context):
            raise RuntimeError("kaboom")

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.boom",
                kwargs={},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs={})

            result = execute_workflow(payload, ctx, lambda name: get(name))

            assert result.success is False
            assert "kaboom" in result.error_message

    def test_manifest_written_after_execution(self):
        @register_workflow(name="test.manifest")
        def with_output(context):
            context.save_json("data.json", {"x": 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.manifest",
                kwargs={},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs={})

            execute_workflow(payload, ctx, lambda name: get(name))

            manifest = ctx.storage.read_json("manifest.json")
            assert "data.json" in manifest["files"]

    def test_parameters_none_when_not_defined(self):
        @register_workflow(name="test.noparam")
        def noparam(context):
            # For non-validated workflows, kwargs returns the raw dict
            assert context.kwargs == {}
            context.save_json("ok.json", {"ok": True})

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = ExecutionPayload(
                workflow_name="test.noparam",
                kwargs={},
                storage_prefix=tmpdir,
            )
            ctx = create_local_context(path=tmpdir, kwargs={})
            result = execute_workflow(payload, ctx, lambda name: get(name))
            assert result.success is True
