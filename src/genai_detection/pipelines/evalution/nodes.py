from kedro.pipeline import Pipeline, pipeline, node

from torchvision import transforms, models
import torch

from genai_detection.settings import TRAIN_CONFIG
from utils.training import create_run, log_metrics, save_checkpoint
from utils.training import create_run

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


def get_model():
    """Get the trained ResNet18 model with the final layer replaced for binary classification."""
    # TODO: fix the model loading
    model = None
    return model


model_loading_node = node(
                func=get_model,
                inputs=None,
                outputs="model",
                name="model_loading",
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


def evaluate_model(model, dataset, wandb_run):
    """Evaluaten the model on the given dataset based on config yaml."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return ''


evaluate_model_node = node(
                func=evaluate_model,
                inputs=["model", "preprocessed_data", "wandb_run"],
                outputs="trained_model",
                name="model_training",
)


def finish_wandb_run(wandb_run) -> None:
    wandb_run.finish()

finish_wandb_run_node = node(
                func=finish_wandb_run,
                inputs="finished_wandb_run",
                outputs=None,
                name="wandb_run_finishing",
)