from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module
import wandb
from wandb.sdk.wandb_run import Run

from utils.config import WANDB_CONFIG, TRAIN_CONFIG


def create_run(config: dict) -> Run:
    """
    Creates a new run on the Weights & Biases service.

    Parameters:
    project_name (str): The name of the project.
    experiment_name (str): The name of the experiment.
    run_name (Optional[str]): The name of the run.
    config (dict): The configuration parameters for the run.

    Returns:
    run (wandb.Run): The run instance.

    Examples:
    >>> run = create_run(
    ...     project_name="training",
    ...     experiment_name="deep-fake-detection",
    ...     run_name="run-1",
    ...     config={
    ...         "model": "ResNet50",
    ...         "batch_size": 32,
    ...         "learning_rate": 0.001,
    ...         "num_epochs": 10,
    ...     },
    ... )
    """
    init_params = {
        "entity": WANDB_CONFIG["entity"],
        "project": WANDB_CONFIG["project"],
        "config": config,
    }
    run = wandb.init(**init_params)
    return run


def log_metrics(run: Run, metrics: dict, step: int, prefix: str = ""):
    """
    Logs the metrics to the Weights & Biases service.

    Parameters:
    run (wandb.Run): The run instance to which the metrics will be logged.
    metrics (dict): A dictionary where the keys are the names of the metrics and the values are the metric values.
        e.g. {"loss": 0.5, "accuracy": 0.9}
    step (int): The current step or epoch in the training loop.
    prefix (str, optional): A string that is prepended to the metric name. Defaults to "".
    """
    run.log({f"{prefix}{key}": value for key, value in metrics.items()}, step=step)


def save_checkpoint(run: Run, model: Module, model_name: str):
    """
    Saves the model's state dict to a file and logs the file as an artifact to the Weights & Biases service.

    Parameters:
    run (wandb.Run): The run instance to which the model artifact will be logged.
    model (torch.nn.Module): The PyTorch model to be saved.
    model_name (str): The name of the model. The model will be saved as 'models/{model_name}.pt'.
    """
    dir_path = Path("models")
    dir_path.mkdir(parents=True, exist_ok=True)
    model_path = dir_path / f'{model_name}.pt'
    torch.save(model, model_path)
    run.link_model(path=model_path, registered_model_name=TRAIN_CONFIG["model_name"])

def download_checkpoint(run: Run, model_name: str):
    """
    Downloads the model's state dict from the Weights & Biases service and loads it into the model.

    Parameters:
    run (wandb.Run): The run instance from which the model artifact will be downloaded.
    model_name (str): The name of the model. The model will be downloaded as 'models/{model_name}.pt'.
    """
    dir_path = Path("models")
    dir_path.mkdir(parents=True, exist_ok=True)
    model_path = dir_path / f'{model_name}.pt'
    artifact = run.use_artifact(f"{run.project}/{run.name}:model")
    artifact.download(root=str(dir_path))
    model = torch.load(model_path)
    return model


def load_checkpoint(model: Module, model_name: str):
    """
    Loads the model's state dict from a file.

    Parameters:
    model (torch.nn.Module): The PyTorch model to which the state dict will be loaded.
    model_name (str): The name of the model. The model will be loaded from 'models/{model_name}.pt'.
    """
    model.load_state_dict(torch.load(f'models/{model_name}.pt'))


def evaluate_model(model: Module, dataloader, criterion):
    """
    Evaluates the model's performance on a dataset.

    Parameters:
    model (torch.nn.Module): The PyTorch model to be evaluated.
    dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset on which the model will be evaluated.
    criterion (torch.nn.modules.loss._Loss): The loss function.

    Returns:
    total_loss (float): The total loss of the model on the dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss
