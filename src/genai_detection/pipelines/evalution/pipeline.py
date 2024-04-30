from kedro.pipeline import Pipeline, pipeline
from genai_detection.pipelines.evaluation.nodes import (
    initialize_wandb_run_node,
    evaluate_model_node,
    model_loading_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            model_loading_node,
            initialize_wandb_run_node,
            evaluate_model_node,
        ],
        inputs=None,
        outputs=None,
        parameters=None,
        namespace=None,
        tags=None,
    )
