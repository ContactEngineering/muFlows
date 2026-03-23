"""Tests for WorkflowImplementation base class."""

from typing import Union

import pytest

from muflows import LocalFolderContext, WorkflowContext, WorkflowImplementation


class SimpleWorkflow(WorkflowImplementation):
    """Simple test workflow."""

    class Meta:
        name = "test.simple_workflow"
        display_name = "Simple Workflow"

    class Parameters(WorkflowImplementation.Parameters):
        multiplier: float = 1.0

    def execute(self, context: WorkflowContext) -> dict:
        multiplier = context.kwargs.get("multiplier", 1.0)
        result = {"value": 42 * multiplier}
        context.save_json("result.json", result)
        return result


class WorkflowWithOptionalParams(WorkflowImplementation):
    """Workflow with optional parameters."""

    class Meta:
        name = "test.optional_params"
        display_name = "Optional Params"

    class Parameters(WorkflowImplementation.Parameters):
        required_param: str
        optional_param: int = 10
        another_optional: Union[str, None] = None

    def execute(self, context: WorkflowContext) -> dict:
        return {
            "required": context.kwargs["required_param"],
            "optional": context.kwargs.get("optional_param", 10),
            "another": context.kwargs.get("another_optional"),
        }


class TestWorkflowImplementation:
    """Tests for WorkflowImplementation base class."""

    def test_meta_attributes(self):
        """Test that Meta attributes are accessible."""
        assert SimpleWorkflow.Meta.name == "test.simple_workflow"
        assert SimpleWorkflow.Meta.display_name == "Simple Workflow"

    def test_execute_with_context(self, tmp_path):
        """Test executing a workflow with a context."""
        context = LocalFolderContext(
            path=tmp_path,
            kwargs={"multiplier": 2.0},
        )

        workflow = SimpleWorkflow()
        result = workflow.execute(context)

        assert result == {"value": 84.0}

        # Verify file was saved
        saved = context.read_json("result.json")
        assert saved == {"value": 84.0}

    def test_execute_with_default_kwargs(self, tmp_path):
        """Test executing with default parameter values."""
        context = LocalFolderContext(
            path=tmp_path,
            kwargs={},
        )

        workflow = SimpleWorkflow()
        result = workflow.execute(context)

        assert result == {"value": 42.0}

    def test_clean_kwargs_validates(self):
        """Test that clean_kwargs validates parameters."""
        # Valid kwargs
        cleaned = SimpleWorkflow.clean_kwargs({"multiplier": 2.5})
        assert cleaned == {"multiplier": 2.5}

        # Invalid extra parameter should raise
        with pytest.raises(Exception):  # pydantic.ValidationError
            SimpleWorkflow.clean_kwargs({"multiplier": 2.5, "invalid": "param"})

    def test_clean_kwargs_fills_defaults(self):
        """Test that clean_kwargs fills default values."""
        cleaned = SimpleWorkflow.clean_kwargs(None)
        assert cleaned == {"multiplier": 1.0}

        cleaned = SimpleWorkflow.clean_kwargs({})
        assert cleaned == {"multiplier": 1.0}

    def test_clean_kwargs_no_fill(self):
        """Test clean_kwargs with fill_missing=False."""
        cleaned = SimpleWorkflow.clean_kwargs(None, fill_missing=False)
        assert cleaned == {}

        cleaned = SimpleWorkflow.clean_kwargs({}, fill_missing=False)
        assert cleaned == {}

    def test_clean_kwargs_with_required_params(self):
        """Test clean_kwargs with required parameters."""
        # Missing required param should raise
        with pytest.raises(Exception):  # pydantic.ValidationError
            WorkflowWithOptionalParams.clean_kwargs({})

        # With required param, fills optional defaults
        cleaned = WorkflowWithOptionalParams.clean_kwargs({"required_param": "test"})
        assert cleaned == {
            "required_param": "test",
            "optional_param": 10,
            "another_optional": None,
        }

    def test_execute_raises_not_implemented(self):
        """Test that base WorkflowImplementation.execute() raises NotImplementedError."""
        workflow = WorkflowImplementation()
        with pytest.raises(NotImplementedError):
            workflow.execute(None)


class TestWorkflowIntegration:
    """Integration tests for workflows with LocalFolderContext."""

    def test_workflow_reads_dependency(self, tmp_path):
        """Test workflow that reads from a dependency."""
        # Create dependency output
        dep_path = tmp_path / "dependency"
        dep_path.mkdir()
        dep_context = LocalFolderContext(
            path=dep_path,
            kwargs={},
        )
        dep_context.save_json("features.json", {"feature1": [1, 2, 3]})

        class DependentWorkflow(WorkflowImplementation):
            class Meta:
                name = "test.dependent"
                display_name = "Dependent"

            def execute(self, context: WorkflowContext) -> dict:
                dep = context.dependency("features")
                features = dep.read_json("features.json")
                return {"sum": sum(features["feature1"])}

        # Create main context with dependency
        main_path = tmp_path / "main"
        main_path.mkdir()
        context = LocalFolderContext(
            path=main_path,
            kwargs={},
            dependency_paths={"features": str(dep_path)},
        )

        workflow = DependentWorkflow()
        result = workflow.execute(context)

        assert result == {"sum": 6}

    def test_workflow_reports_progress(self, tmp_path, capsys):
        """Test workflow that reports progress."""

        class ProgressWorkflow(WorkflowImplementation):
            class Meta:
                name = "test.progress"
                display_name = "Progress"

            def execute(self, context: WorkflowContext) -> dict:
                for i in range(3):
                    context.report_progress(i + 1, 3, f"Step {i + 1}")
                return {"steps": 3}

        context = LocalFolderContext(
            path=tmp_path,
            kwargs={},
        )

        workflow = ProgressWorkflow()
        result = workflow.execute(context)

        assert result == {"steps": 3}

        # LocalFolderContext prints progress
        captured = capsys.readouterr()
        assert "33.3%" in captured.out
        assert "66.7%" in captured.out
        assert "100.0%" in captured.out
