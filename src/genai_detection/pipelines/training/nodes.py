from kedro.pipeline import Pipeline, pipeline, node

from torchvision import transforms, models
import torch

from genai_detection.settings import TRAIN_CONFIG
from utils.training import create_run, log_metrics, save_checkpoint
from utils.model import GenAIDetectorModel

from datetime import datetime

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
                inputs="image_train_data",
                outputs="preprocessed_data",
                name="raw_image_data_preprocessing",
            )


def load_config():
    """Load the training configuration."""
    return TRAIN_CONFIG


load_config_node = node(
                func=load_config,
                inputs=None,
                outputs="config",
                name="config_loading",
)


def get_model():
    """Get a pre-trained ResNet18 model with the final layer replaced for binary classification."""
    model = GenAIDetectorModel()
    return model


model_initialization_node = node(
                func=get_model,
                inputs=None,
                outputs="model",
                name="model_initialization",
)


def initialize_wandb_run(config):
    """Initialize a W&B run."""
    return create_run(config=config)


initialize_wandb_run_node = node(
                func=initialize_wandb_run,
                inputs="config",
                outputs="wandb_run",
                name="wandb_run_initialization",
)


def train_model(model, dataset, config, wandb_run):
    """Train the model on the given dataset based on config yaml."""
    batch_size = config["batch_size"]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(config["epochs"]):
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config["learning_rate"],
                momentum=config["momentum"],
                )
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(outputs, labels.reshape(-1, 1).to(torch.float))
            loss.backward()
            optimizer.step()
            log_metrics(wandb_run,
                        {
                            "loss": loss.item(),
                            "epoch": epoch,
                        },
                        step=(epoch+1)*(i+1))

    return model


train_model_node = node(
                func=train_model,
                inputs=["model", "preprocessed_data", "config", "wandb_run"],
                outputs="trained_model",
                name="model_training",
)


def save_and_upload_model(model, wandb_run, config):
    """Save the trained model and upload it to W&B."""
    save_checkpoint(wandb_run, model, config['model_name'])
    return wandb_run

save_and_upload_model_node = node(
                func=save_and_upload_model,
                inputs=["trained_model", "wandb_run", "config"],
                outputs="finished_wandb_run",
                name="model_saving_and_uploading",
)

def finish_wandb_run(wandb_run) -> None:
    wandb_run.finish()

finish_wandb_run_node = node(
                func=finish_wandb_run,
                inputs="finished_wandb_run",
                outputs=None,
                name="wandb_run_finishing",
)