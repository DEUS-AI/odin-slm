#!/usr/bin/env python3
"""Backfill all existing experiments into MLflow Docker server.

Reads configs, trainer states, and evaluation results from disk and
registers them as MLflow runs so the full experiment history is available
in the MLflow UI.

Usage:
    python scripts/backfill_mlflow.py
"""

import json
import sys
from pathlib import Path

import mlflow
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from mlflow_config import setup_mlflow


# Map experiment dirs to config files and metadata
EXPERIMENTS = [
    {
        "name": "v1",
        "experiment_dir": "experiments/medical_ner_re",
        "config": "configs/medical_ner_re_config.yaml",
        "description": "V1 baseline - synthetic data, Llama 3.2 1B Instruct",
    },
    {
        "name": "v2",
        "experiment_dir": "experiments/medical_ner_re_v2",
        "config": "configs/medical_ner_re_config_v2.yaml",
        "description": "V2 - synthetic data v2, more entity/relation types",
    },
    {
        "name": "v3",
        "experiment_dir": "experiments/medical_ner_re_v3",
        "config": "configs/medical_ner_re_config_v3_stage1.yaml",
        "description": "V3 - two-stage training (stage 1: entities only)",
    },
    {
        "name": "v4",
        "experiment_dir": "experiments/medical_ner_re_v4",
        "config": "configs/medical_ner_re_config_v4.yaml",
        "description": "V4 - entities-only training",
    },
    {
        "name": "v5",
        "experiment_dir": "experiments/medical_ner_re_v5",
        "config": "configs/medical_ner_re_config_v5.yaml",
        "description": "V5 - instruction tuning adjustments",
    },
    {
        "name": "v6",
        "experiment_dir": "experiments/medical_ner_re_v6",
        "config": "configs/medical_ner_re_config_v6.yaml",
        "description": "V6 - standard instruction tuning",
    },
    {
        "name": "v7",
        "experiment_dir": "experiments/medical_ner_re_v7",
        "config": "configs/medical_ner_re_config_v7.yaml",
        "description": "V7 - output-only training fix",
    },
    {
        "name": "v9",
        "experiment_dir": "experiments/medical_ner_re_v9",
        "config": "configs/medical_ner_re_config_v9.yaml",
        "description": "V9 - new baseline with Llama 3.1 8B Base",
    },
    {
        "name": "v10",
        "experiment_dir": "experiments/medical_ner_re_v10",
        "config": "configs/medical_ner_re_config_v10.yaml",
        "description": "V10 - removed repetition penalty, greedy decoding",
    },
    {
        "name": "v11",
        "experiment_dir": "experiments/medical_ner_re_v11",
        "config": "configs/medical_ner_re_config_v11.yaml",
        "description": "V11 - hyperparameter tuning",
    },
    {
        "name": "v12",
        "experiment_dir": "experiments/medical_ner_re_v12",
        "config": "configs/medical_ner_re_config_v12.yaml",
        "description": "V12 - continued tuning",
    },
    {
        "name": "v12b",
        "experiment_dir": "experiments/medical_ner_re_v12b",
        "config": "configs/medical_ner_re_config_v12b.yaml",
        "description": "V12b - first real data (ADE Corpus V2)",
    },
    {
        "name": "v13",
        "experiment_dir": "experiments/medical_ner_re_v13",
        "config": "configs/medical_ner_re_config_v13.yaml",
        "description": "V13 - ADE-only, 5 epochs, best model (Entity F1=0.918)",
    },
    {
        "name": "v14",
        "experiment_dir": "experiments/medical_ner_re_v14",
        "config": "configs/medical_ner_re_config_v14.yaml",
        "description": "V14 - combined real+synthetic, 2 epochs, max_seq=512",
    },
    {
        "name": "v15",
        "experiment_dir": "experiments/medical_ner_re_v15",
        "config": "configs/medical_ner_re_config_v15.yaml",
        "description": "V15 - real-only, 3 epochs, max_seq=1024",
    },
]


def load_config(config_path):
    """Load YAML config file."""
    path = project_root / config_path
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def load_eval_results(eval_path):
    """Load evaluation results JSON."""
    if not eval_path.exists():
        return None
    with open(eval_path) as f:
        return json.load(f)


def load_trainer_state(experiment_dir):
    """Load trainer_state.json for training metrics."""
    path = experiment_dir / "trainer_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def backfill_experiment(exp_info):
    """Create an MLflow run for a single experiment."""
    name = exp_info["name"]
    experiment_dir = project_root / exp_info["experiment_dir"]

    if not experiment_dir.exists():
        print(f"  SKIP {name}: experiment dir not found")
        return

    print(f"\n{'='*60}")
    print(f"  Backfilling {name}: {exp_info['description']}")
    print(f"{'='*60}")

    config = load_config(exp_info["config"])
    trainer_state = load_trainer_state(experiment_dir)

    # Find all evaluation result files
    eval_files = list(experiment_dir.glob("evaluation_results*.json"))

    # Create a training run
    with mlflow.start_run(run_name=f"train-{name}"):
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_version", name)
        mlflow.set_tag("description", exp_info["description"])
        mlflow.set_tag("backfilled", "true")

        # Log config params
        if config:
            params = {}
            if "model" in config:
                for k, v in config["model"].items():
                    params[f"model.{k}"] = str(v)
            if "training" in config:
                for k, v in config["training"].items():
                    params[f"training.{k}"] = str(v)
            if "lora" in config:
                for k, v in config["lora"].items():
                    params[f"lora.{k}"] = str(v)
            if "dataset" in config:
                for k, v in config["dataset"].items():
                    params[f"dataset.{k}"] = str(v)
            if params:
                mlflow.log_params(params)
                print(f"  Logged {len(params)} params from config")

            # Log config as artifact
            config_path = project_root / exp_info["config"]
            if config_path.exists():
                mlflow.log_artifact(str(config_path), "configs")

        # Log training metrics from trainer_state
        if trainer_state:
            log_history = trainer_state.get("log_history", [])

            # Get final train loss
            train_losses = [e for e in log_history if "loss" in e and "eval_loss" not in e]
            if train_losses:
                mlflow.log_metric("final_train_loss", train_losses[-1]["loss"])

            # Get best eval loss
            eval_losses = [e for e in log_history if "eval_loss" in e]
            if eval_losses:
                best_eval = min(eval_losses, key=lambda x: x["eval_loss"])
                mlflow.log_metric("best_eval_loss", best_eval["eval_loss"])
                mlflow.log_metric("best_eval_epoch", best_eval.get("epoch", 0))

                # Log all eval losses as step metrics
                for entry in eval_losses:
                    step = entry.get("step", 0)
                    mlflow.log_metric("eval_loss", entry["eval_loss"], step=step)

            # Log total training time
            total_steps = trainer_state.get("global_step", 0)
            mlflow.log_metric("total_steps", total_steps)

            print(f"  Logged training metrics (steps={total_steps})")

    # Create evaluation runs
    for eval_file in eval_files:
        eval_data = load_eval_results(eval_file)
        if not eval_data:
            continue

        # Derive test set name from filename
        suffix = eval_file.stem.replace("evaluation_results", "").strip("_")
        test_set_label = suffix if suffix else "default"

        with mlflow.start_run(run_name=f"eval-{name}-{test_set_label}"):
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("model_version", name)
            mlflow.set_tag("test_set", test_set_label)
            mlflow.set_tag("backfilled", "true")

            # Log entity metrics
            if "entity_metrics" in eval_data:
                micro = eval_data["entity_metrics"].get("micro", {})
                macro = eval_data["entity_metrics"].get("macro", {})
                metrics = {}
                if micro:
                    metrics["entity_f1_micro"] = micro.get("f1", 0)
                    metrics["entity_precision_micro"] = micro.get("precision", 0)
                    metrics["entity_recall_micro"] = micro.get("recall", 0)
                    metrics["entity_tp"] = micro.get("tp", 0)
                    metrics["entity_fp"] = micro.get("fp", 0)
                    metrics["entity_fn"] = micro.get("fn", 0)
                if macro:
                    metrics["entity_f1_macro"] = macro.get("f1", 0)

                mlflow.log_metrics(metrics)

            # Log relation metrics
            if "relation_metrics" in eval_data:
                micro = eval_data["relation_metrics"].get("micro", {})
                macro = eval_data["relation_metrics"].get("macro", {})
                metrics = {}
                if micro:
                    metrics["relation_f1_micro"] = micro.get("f1", 0)
                    metrics["relation_precision_micro"] = micro.get("precision", 0)
                    metrics["relation_recall_micro"] = micro.get("recall", 0)
                    metrics["relation_tp"] = micro.get("tp", 0)
                    metrics["relation_fp"] = micro.get("fp", 0)
                    metrics["relation_fn"] = micro.get("fn", 0)
                if macro:
                    metrics["relation_f1_macro"] = macro.get("f1", 0)

                mlflow.log_metrics(metrics)

            # Log eval results as artifact
            mlflow.log_artifact(str(eval_file), "evaluation")

            entity_f1 = eval_data.get("entity_metrics", {}).get("micro", {}).get("f1", "N/A")
            rel_f1 = eval_data.get("relation_metrics", {}).get("micro", {}).get("f1", "N/A")
            print(f"  Eval ({test_set_label}): Entity F1={entity_f1}, Relation F1={rel_f1}")


def main():
    print("=" * 60)
    print("BACKFILLING EXPERIMENTS TO MLFLOW")
    print(f"Server: {mlflow.get_tracking_uri()}")
    print("=" * 60)

    setup_mlflow()

    for exp in EXPERIMENTS:
        backfill_experiment(exp)

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"\nView results at: http://localhost:5000")


if __name__ == "__main__":
    main()
