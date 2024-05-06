from kedro.pipeline import Pipeline, pipeline
from genai_detection.pipelines.evaluation.nodes import (
    initialize_eval_wandb_run_node,
    get_data_loader_node,
    model_loading_node,
    evaluate_predictions_node,
    preprocess_raw_image_data_node,
    make_predictions_node,
    log_metrics_node,
    finish_wandb_run_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            initialize_eval_wandb_run_node,
            preprocess_raw_image_data_node,
            get_data_loader_node,
            model_loading_node,
            evaluate_predictions_node,
            make_predictions_node,
            log_metrics_node,
            finish_wandb_run_node,
        ],
        inputs=None,
        outputs=None,
        parameters=None,
        namespace=None,
        tags=None,
    )
