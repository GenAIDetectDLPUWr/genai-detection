'''Utilities for model download and loading'''

import wandb
from wandb.sdk.wandb_run import Run
from pathlib import Path
import os

from training import create_run
from config import API_CONFIG

def download_model_from_reqistry(run: Run, model_name: str, version: str = "latest", overwrite: bool = False):
    """
    Downloads the model's state dict from the Weights & Biases service and loads it into the model.

    Parameters:
    run (wandb.Run): The run instance from which the model artifact will be downloaded.
    model_name (str): The name of the model. The model will be downloaded as 'models/{model_name}.pt'.
    version (str): The version of the model to download. Defaults to "latest".
    overwrite (bool): If True, the model file will be overwritten if it already exists. Defaults to False.
    """

    destination_path = Path(API_CONFIG["model_path"]) / f"{model_name}.pt"

    if not overwrite and destination_path.exists():
        # Possibly change to a logger
        print(f"File already exists at {destination_path}")
    else:
        downloaded_model_path = Path(run.use_model(name=f"{model_name}:{version}"))
        os.rename(downloaded_model_path, destination_path)

    return destination_path
