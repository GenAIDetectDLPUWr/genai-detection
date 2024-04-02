import logging
from kedro.pipeline import Pipeline, pipeline, node

from torchvision import transforms, models
import torch

LOGGER = logging.getLogger(__name__)

def transform_raw_image_data(raw_image_dataset):
    """Preprocess raw image data."""
    transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    raw_image_dataset.transform = transformation
    return raw_image_dataset

preprocess_raw_image_data_node = node(
                func=transform_raw_image_data,
                inputs="image_test_data",
                outputs="preprocessed_data",
                name="raw_image_data_preprocessing",
            )

def get_model():
    model = models.resnet18(weights=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 1)
    return model

model_initialization_node = node(
                func=get_model,
                inputs=None,
                outputs="model",
                name="model_initialization",
)

def train_model(model, dataset):
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(2):
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(outputs, labels.reshape(-1, 1).to(torch.float))
            loss.backward()
            optimizer.step()
            LOGGER.info(f"Epoch: {epoch}, Loss: {loss.item()}")

    return model

train_model_node = node(
                func=train_model,
                inputs=["model", "preprocessed_data"],
                outputs="trained_model",
                name="model_training",
)
