from genai_detection.pipelines.training.nodes import transform_raw_image_data, get_model
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch
import pytest
from genai_detection.settings import TRAIN_CONFIG as config


@pytest.fixture
def test_model():
    return get_model()


@pytest.fixture
def test_image_folder():
    return ImageFolder(
        root="src/tests/data/test_dataset",
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    )


def test_transform_raw_image_data(test_image_folder):
    transformed_data = transform_raw_image_data(test_image_folder)
    output = str(transformed_data)

    target_str_representation = """Dataset ImageFolder
    Number of datapoints: 20
    Root location: src/tests/data/test_dataset
    StandardTransform
Transform: Compose(
               Resize(size=(256, 256), interpolation=bilinear, max_size=None, antialias=True)
               ToTensor()
           )"""
    assert output == target_str_representation


def test_model_output_shape(test_image_folder, test_model):
    dataloader = torch.utils.data.DataLoader(test_image_folder, batch_size=config["batch_size"], shuffle=True)
    for data in dataloader:
        inputs, labels = data
        optimizer = torch.optim.SGD(
            test_model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            )
        optimizer.zero_grad()
        outputs = torch.sigmoid(test_model(inputs))
        assert outputs.shape == (labels.shape[0], 1)