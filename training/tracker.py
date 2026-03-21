"""MLflow experiment tracking."""


def log_experiment(params: dict, metrics: dict) -> str:
    """Log an experiment run to MLflow. Return run_id."""
    raise NotImplementedError
