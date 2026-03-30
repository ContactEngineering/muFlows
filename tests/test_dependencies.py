"""Tests for workflow dependency specifications."""

import pytest

from muflow.dependencies import (
    WorkflowSpec,
    enumerate_specs,
)


class TestWorkflowSpec:
    """Tests for WorkflowSpec dataclass."""

    def test_basic_creation(self):
        """Should create a spec with required fields."""
        spec = WorkflowSpec(workflow="myapp.compute")
        assert spec.workflow == "myapp.compute"
        assert spec.subject_key is None
        assert spec.kwargs is None
        assert spec.key is None

    def test_full_creation(self):
        """Should create a spec with all fields."""
        spec = WorkflowSpec(
            workflow="myapp.compute",
            subject_key="surface:123",
            kwargs={"param": "value"},
            key="my_dep",
        )
        assert spec.workflow == "myapp.compute"
        assert spec.subject_key == "surface:123"
        assert spec.kwargs == {"param": "value"}
        assert spec.key == "my_dep"

    def test_with_defaults_applies_subject(self):
        """with_defaults should apply default subject_key."""
        spec = WorkflowSpec(workflow="myapp.compute")
        filled = spec.with_defaults("tag:1", {"k": "v"})

        assert filled.subject_key == "tag:1"
        assert filled.kwargs == {"k": "v"}

    def test_with_defaults_preserves_explicit(self):
        """with_defaults should not override explicit values."""
        spec = WorkflowSpec(
            workflow="myapp.compute",
            subject_key="surface:999",
            kwargs={"explicit": True},
        )
        filled = spec.with_defaults("tag:1", {"default": True})

        assert filled.subject_key == "surface:999"
        assert filled.kwargs == {"explicit": True}


class TestEnumerateSpecs:
    """Tests for enumerate_specs function."""

    def test_none_returns_empty(self):
        """None declaration should return empty dict."""
        result = enumerate_specs(None, "tag:1", {})
        assert result == {}

    def test_simple_string(self):
        """String value should create spec with inherited subject/kwargs."""
        decl = {"features": "myapp.features"}
        result = enumerate_specs(decl, "tag:1", {"param": "value"})

        assert "features" in result
        assert result["features"].workflow == "myapp.features"
        assert result["features"].subject_key == "tag:1"
        assert result["features"].kwargs == {"param": "value"}

    def test_explicit_spec(self):
        """WorkflowSpec should be used directly with defaults applied."""
        decl = {
            "features": WorkflowSpec(
                workflow="myapp.features",
                subject_key="surface:123",
            )
        }
        result = enumerate_specs(decl, "tag:1", {"param": "value"})

        assert result["features"].workflow == "myapp.features"
        assert result["features"].subject_key == "surface:123"
        # kwargs should use default since not specified
        assert result["features"].kwargs == {"param": "value"}

    def test_callable_enumeration(self):
        """Callable should be invoked to enumerate specs."""

        def enumerate_surfaces(subject_key, kwargs):
            surfaces = kwargs.get("surfaces", [])
            return {
                f"surface_{i}": WorkflowSpec(
                    workflow="myapp.process",
                    subject_key=f"surface:{s['id']}",
                    kwargs={"surface_id": s["id"]},
                )
                for i, s in enumerate(surfaces)
            }

        result = enumerate_specs(
            enumerate_surfaces,
            "tag:1",
            {"surfaces": [{"id": 10}, {"id": 20}, {"id": 30}]},
        )

        assert len(result) == 3
        assert result["surface_0"].subject_key == "surface:10"
        assert result["surface_1"].subject_key == "surface:20"
        assert result["surface_2"].subject_key == "surface:30"

    def test_nested_callable_in_dict(self):
        """Dict with callable values should namespace keys."""

        def enumerate_folds(subject_key, kwargs):
            n_folds = kwargs.get("n_folds", 3)
            return {
                f"{i}": WorkflowSpec(
                    workflow="myapp.fold",
                    kwargs={"fold": i},
                )
                for i in range(n_folds)
            }

        decl = {
            "static": "myapp.static",
            "folds": enumerate_folds,
        }
        result = enumerate_specs(decl, "tag:1", {"n_folds": 3})

        assert "static" in result
        assert "folds.0" in result
        assert "folds.1" in result
        assert "folds.2" in result

    def test_mixed_declaration(self):
        """Dict with mixed types should handle all correctly."""
        decl = {
            "simple": "myapp.simple",
            "explicit": WorkflowSpec(workflow="myapp.explicit", subject_key="s:1"),
        }
        result = enumerate_specs(decl, "tag:1", {"p": 1})

        assert result["simple"].workflow == "myapp.simple"
        assert result["explicit"].workflow == "myapp.explicit"
        assert result["explicit"].subject_key == "s:1"

    def test_invalid_spec_type_raises(self):
        """Invalid spec type should raise TypeError."""
        decl = {"bad": 123}  # Not a valid type

        with pytest.raises(TypeError, match="Invalid dependency spec"):
            enumerate_specs(decl, "tag:1", {})

    def test_invalid_declaration_type_raises(self):
        """Invalid declaration type should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid declaration type"):
            enumerate_specs("not_a_dict_or_callable", "tag:1", {})
