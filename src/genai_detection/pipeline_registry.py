"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from genai_detection.pipelines import evaluation, training


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "training": training.create_pipeline(),
        "evaluation": evaluation.create_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
