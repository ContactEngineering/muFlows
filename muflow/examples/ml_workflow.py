"""ML pipeline example using the Pipeline abstraction.

Demonstrates a three-stage ML workflow with parallelism at every stage:

1. **Compute features** on N datasets (N parallel jobs)
2. **Train model** + **LOO cross-validation** (1 + N parallel jobs)
3. **Generate reports** in PDF, XLSX, CSV (3 parallel jobs)

Usage::

    from muflow.examples.ml_workflow import ml_pipeline

    plan = ml_pipeline.build_plan(
        subject_key="experiment:1",
        kwargs={"datasets": ["dataset_a", "dataset_b", "dataset_c"]},
    )
    backend.submit_plan(plan)
"""

from muflow import register_workflow
from muflow.pipeline import ForEach, Pipeline, Step


# ── Pure workflows (no DAG knowledge) ─────────────────────────────────────


@register_workflow(name="ml.compute_features")
def compute_features(context):
    """Compute features for a single dataset."""
    ds = context.kwargs["dataset_name"]
    # ... load dataset, extract features ...
    context.save_json("features.json", {"dataset": ds, "features": []})


@register_workflow(name="ml.train_model")
def train_model(context):
    """Train model on all feature sets."""
    all_features = []
    for key in context.dependency_keys():
        all_features.append(context.dependency(key).read_json("features.json"))
    # ... train model on all_features ...
    context.save_json("model.json", {
        "weights": [],
        "n_datasets": len(all_features),
    })


@register_workflow(name="ml.loo_cv")
def loo_cv(context):
    """Leave-one-out cross-validation for a single fold."""
    leave_out = context.kwargs["leave_out_index"]
    train_features = []
    for key in context.dependency_keys():
        feat = context.dependency(key).read_json("features.json")
        if key != f"features:{leave_out}":
            train_features.append(feat)
    # ... train on train_features, evaluate on val_features ...
    context.save_json("cv_result.json", {
        "fold": leave_out,
        "score": 0.0,
        "n_train": len(train_features),
    })


@register_workflow(name="ml.generate_report")
def generate_report(context):
    """Generate report in a single format (pdf/xlsx/csv)."""
    fmt = context.kwargs["format"]
    context.dependency("train").read_json("model.json")  # ensure train is done
    cv_results = []
    for key in context.dependency_keys():
        if key.startswith("loo_cv:"):
            cv_results.append(
                context.dependency(key).read_json("cv_result.json")
            )
    # ... generate report ...
    context.save_file(f"report.{fmt}", b"...")


# ── Pipeline definition (DAG topology in one place) ───────────────────────


ml_pipeline = Pipeline(
    name="ml.full_pipeline",
    display_name="ML Training Pipeline",
    steps={
        "features": ForEach(
            workflow="ml.compute_features",
            over=lambda subject_key, kwargs: [
                {"dataset_name": ds} for ds in kwargs["datasets"]
            ],
        ),
        "train": Step(
            workflow="ml.train_model",
            after=["features"],
        ),
        "loo_cv": ForEach(
            workflow="ml.loo_cv",
            after=["features"],
            over=lambda subject_key, kwargs: [
                {"leave_out_index": i, "datasets": kwargs["datasets"]}
                for i in range(len(kwargs["datasets"]))
            ],
        ),
        "reports": ForEach(
            workflow="ml.generate_report",
            after=["train", "loo_cv"],
            over=lambda subject_key, kwargs: [
                {"format": fmt} for fmt in ("pdf", "xlsx", "csv")
            ],
        ),
    },
)
