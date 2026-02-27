#!/usr/bin/env python3
"""Register all trained models in MLflow Model Registry.

Uses mlflow.pyfunc.log_model() to create proper LoggedModel entries
(required by MLflow v3) and registers them in the Model Registry with
full traceability to training experiments.

Only small config files are uploaded as artifacts (not full weights).
Model weights remain on disk at their experiment directory paths.

Usage:
    python scripts/register_models_mlflow.py
"""

import json
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from mlflow_config import setup_mlflow

REGISTERED_MODEL_NAME = "odin-medical-ner-re"


class LoRAAdapterModel(mlflow.pyfunc.PythonModel):
    """Lightweight wrapper for LoRA adapter models.

    This is a metadata-only model for registry traceability.
    Actual inference uses Unsloth to load the adapter from disk.
    """

    def load_context(self, context):
        self.model_info = context.artifacts.get("model_info", {})

    def predict(self, context, model_input):
        raise NotImplementedError(
            "Load this model using Unsloth/PEFT, not MLflow predict. "
            "See adapter_config.json for LoRA configuration."
        )


# Map experiment versions to their info
MODELS = [
    {
        "version": "v1",
        "experiment_dir": "experiments/medical_ner_re",
        "description": "V1 baseline - synthetic data, Llama 3.2 1B Instruct, LoRA r=16",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "data": "synthetic_v1",
        "tags": {"data_type": "synthetic", "base_model_size": "1B"},
    },
    {
        "version": "v2",
        "experiment_dir": "experiments/medical_ner_re_v2",
        "description": "V2 - synthetic data v2, 5 entity types, 4 relation types",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "1B"},
    },
    {
        "version": "v4",
        "experiment_dir": "experiments/medical_ner_re_v4",
        "description": "V4 - entities-only training",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "data": "synthetic_v2_entities_only",
        "tags": {"data_type": "synthetic", "base_model_size": "1B"},
    },
    {
        "version": "v5",
        "experiment_dir": "experiments/medical_ner_re_v5",
        "description": "V5 - instruction tuning adjustments",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "1B"},
    },
    {
        "version": "v6",
        "experiment_dir": "experiments/medical_ner_re_v6",
        "description": "V6 - standard instruction tuning",
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "1B"},
    },
    {
        "version": "v9",
        "experiment_dir": "experiments/medical_ner_re_v9",
        "description": "V9 - new baseline with Llama 3.1 8B Base",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "8B"},
    },
    {
        "version": "v10",
        "experiment_dir": "experiments/medical_ner_re_v10",
        "description": "V10 - removed repetition penalty, greedy decoding",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "8B"},
    },
    {
        "version": "v11",
        "experiment_dir": "experiments/medical_ner_re_v11",
        "description": "V11 - hyperparameter tuning",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "8B"},
    },
    {
        "version": "v12",
        "experiment_dir": "experiments/medical_ner_re_v12",
        "description": "V12 - continued tuning",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "synthetic_v2",
        "tags": {"data_type": "synthetic", "base_model_size": "8B"},
    },
    {
        "version": "v12b",
        "experiment_dir": "experiments/medical_ner_re_v12b",
        "description": "V12b - first real data (ADE Corpus V2)",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "ade_corpus_v2",
        "tags": {"data_type": "real", "base_model_size": "8B"},
    },
    {
        "version": "v13",
        "experiment_dir": "experiments/medical_ner_re_v13",
        "description": "V13 - BEST MODEL. ADE-only, 5 epochs. Entity F1=0.918, Relation F1=0.842",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "ade_corpus_v2",
        "tags": {"data_type": "real", "base_model_size": "8B", "best_model": "true"},
    },
    {
        "version": "v14",
        "experiment_dir": "experiments/medical_ner_re_v14",
        "description": "V14 - combined real+synthetic, 2 epochs, max_seq=512. Entity F1=0.911, Relation F1=0.832",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "combined_ade_bc5cdr_biored_synthetic",
        "tags": {"data_type": "mixed", "base_model_size": "8B"},
    },
    {
        "version": "v15",
        "experiment_dir": "experiments/medical_ner_re_v15",
        "description": "V15 - real-only, 3 epochs, max_seq=1024. Entity F1=0.825, Relation F1=0.711",
        "base_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "data": "combined_ade_bc5cdr_biored",
        "tags": {"data_type": "real", "base_model_size": "8B"},
    },
]


# Small config files to upload (not the full weights)
CONFIG_FILES = [
    "adapter_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
]


def get_eval_metrics(experiment_dir):
    """Load evaluation metrics from results file."""
    eval_path = experiment_dir / "evaluation_results.json"
    if not eval_path.exists():
        return {}
    with open(eval_path) as f:
        data = json.load(f)
    metrics = {}
    if "entity_metrics" in data and "micro" in data["entity_metrics"]:
        metrics["entity_f1"] = data["entity_metrics"]["micro"].get("f1", 0)
    if "relation_metrics" in data and "micro" in data["relation_metrics"]:
        metrics["relation_f1"] = data["relation_metrics"]["micro"].get("f1", 0)
    return metrics


def register_model(model_info, client, experiment_id):
    """Log model with pyfunc and register in Model Registry."""
    version = model_info["version"]
    experiment_dir = project_root / model_info["experiment_dir"]
    model_dir = experiment_dir / "final_model"

    if not model_dir.exists():
        print(f"  SKIP {version}: no final_model directory")
        return

    print(f"\n  Registering {version}...")

    # Find the existing training run for this version
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_version = '{version}' AND tags.stage = 'training'",
    )

    if not runs:
        print(f"    WARNING: No training run found for {version}, creating new one")
        with mlflow.start_run(run_name=f"train-{version}") as run:
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("model_version", version)
            run_id = run.info.run_id
    else:
        run_id = runs[0].info.run_id

    # Collect small config files as artifacts dict
    artifacts = {}
    for cfg_file in CONFIG_FILES:
        cfg_path = model_dir / cfg_file
        if cfg_path.exists():
            artifacts[cfg_file.replace(".", "_")] = str(cfg_path)

    # Log model using pyfunc (creates LoggedModel entry required by MLflow v3)
    with mlflow.start_run(run_id=run_id):
        # Add model metadata tags
        for key, value in model_info["tags"].items():
            mlflow.set_tag(key, value)
        mlflow.set_tag("base_model", model_info["base_model"])
        mlflow.set_tag("training_data", model_info["data"])
        mlflow.set_tag("model_local_path", str(model_dir))

        # Log the model with pyfunc wrapper + small config artifacts
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LoRAAdapterModel(),
            artifacts=artifacts if artifacts else None,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    # Get the latest version just created
    model_versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )
    latest_mv = max(model_versions, key=lambda v: int(v.version))

    # Update version description
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=latest_mv.version,
        description=model_info["description"],
    )

    # Set alias matching our version name
    # MLflow v3 reserves "v<number>" aliases, so prefix with "model-"
    alias_name = f"model-{version}"
    try:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=alias_name,
            version=latest_mv.version,
        )
    except Exception as e:
        print(f"    WARNING: Could not set alias '{alias_name}': {e}")

    # Set champion alias for the best model
    if model_info["tags"].get("best_model") == "true":
        try:
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="champion",
                version=latest_mv.version,
            )
            print(f"    Set 'champion' alias")
        except Exception as e:
            print(f"    WARNING: Could not set champion alias: {e}")

    eval_metrics = get_eval_metrics(experiment_dir)
    ef1 = eval_metrics.get("entity_f1", "N/A")
    rf1 = eval_metrics.get("relation_f1", "N/A")
    if isinstance(ef1, float):
        ef1 = f"{ef1:.3f}"
    if isinstance(rf1, float):
        rf1 = f"{rf1:.3f}"
    print(f"    Registry version {latest_mv.version} | alias: {version} | Entity F1={ef1}, Rel F1={rf1}")


def main():
    setup_mlflow()

    print("=" * 60)
    print("REGISTERING MODELS IN MLFLOW MODEL REGISTRY")
    print(f"Server: {mlflow.get_tracking_uri()}")
    print(f"Model name: {REGISTERED_MODEL_NAME}")
    print("=" * 60)

    client = mlflow.MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name("odin-slm-medical-ner")
    if experiment is None:
        print("ERROR: Experiment not found. Run backfill_mlflow.py first.")
        sys.exit(1)

    for model_info in MODELS:
        try:
            register_model(model_info, client, experiment.experiment_id)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("MODEL REGISTRY SUMMARY")
    print("=" * 60)

    model_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    for mv in sorted(model_versions, key=lambda v: int(v.version)):
        aliases = mv.aliases if hasattr(mv, "aliases") else []
        alias_str = f" [{', '.join(aliases)}]" if aliases else ""
        desc = mv.description[:65] if mv.description else "N/A"
        print(f"  v{mv.version:>2s}{alias_str}: {desc}")

    print(f"\nTotal: {len(model_versions)} model versions")
    print(f"View at: http://localhost:5000/#/models/{REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()
