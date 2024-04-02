import logging
from kedro.pipeline import Pipeline, pipeline, node

LOGGER = logging.getLogger(__name__)

def transform_raw_image_data(raw_image_data):
    return raw_image_data

preprocess_raw_image_data_node = node(
                func=transform_raw_image_data,
                inputs="image_test_data",
                outputs="preprocessed_data",
                name="raw_image_data_preprocessing",
            )