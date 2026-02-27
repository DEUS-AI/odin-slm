"""MLflow configuration for Odin SLM experiment tracking

This module provides configuration and utility functions for MLflow experiment tracking.
All training and evaluation runs are automatically logged to MLflow for reproducibility.

MLflow server runs in Docker (see docker-compose.yml).
"""

import mlflow
import os
from pathlib import Path

# MLflow tracking configuration - Docker server
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def setup_mlflow(experiment_name="odin-slm-medical-ner"):
    """Initialize MLflow tracking

    Connects to the MLflow tracking server (Docker-based).

    Args:
        experiment_name: Name of the MLflow experiment to use

    Returns:
        mlflow module configured for tracking
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Get or create experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)

    # Set the experiment as active
    mlflow.set_experiment(experiment_name)

    return mlflow


def get_git_commit_hash():
    """Get current git commit hash for reproducibility

    Returns:
        str: Current git commit SHA, or "unknown" if git is not available
    """
    import subprocess
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch():
    """Get current git branch name

    Returns:
        str: Current git branch, or "unknown" if git is not available
    """
    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_system_info(mlflow_instance):
    """Log system information to MLflow

    Args:
        mlflow_instance: Active MLflow instance
    """
    import platform
    import torch

    # System info
    mlflow_instance.set_tag("os", platform.system())
    mlflow_instance.set_tag("python_version", platform.python_version())

    # PyTorch info
    mlflow_instance.set_tag("pytorch_version", torch.__version__)
    mlflow_instance.set_tag("cuda_available", torch.cuda.is_available())

    if torch.cuda.is_available():
        mlflow_instance.set_tag("cuda_version", torch.version.cuda)
        mlflow_instance.set_tag("gpu_name", torch.cuda.get_device_name(0))
        mlflow_instance.set_tag("gpu_count", torch.cuda.device_count())
