from pathlib import Path

import pytest
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

from genai_detection.pipelines.data_processing.nodes import transform_raw_image_data
from torchvision import transforms

def test_transform_raw_image_data(test_image_folder):
    transformed_data = transform_raw_image_data(test_image_folder)
    assert transformed_data.transform[0] == transforms.Resize((256, 256))
    assert transformed_data.transform.transforms[1].__class__.__name__ == 'ToTensor'