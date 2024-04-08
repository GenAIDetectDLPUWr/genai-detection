from kedro.pipeline import Pipeline, pipeline
from genai_detection.pipelines.data_processing.nodes import (
    preprocess_raw_image_data_node,
    model_initialization_node,
    train_model_node,
    load_config_node,
    initialize_wandb_run_node,
    save_and_upload_model_node
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            load_config_node,
            initialize_wandb_run_node,
            preprocess_raw_image_data_node,
            model_initialization_node,
            train_model_node,
            save_and_upload_model_node,
        ],
        inputs=None,
        outputs=None,
        parameters=None,
        namespace=None,
        tags=None,
    )
