"""Utilities for model download and loading"""

import wandb
from wandb.sdk.wandb_run import Run
from pathlib import Path
import os
import torch
import numpy as np
from torchvision import models, transforms

from genai_utils.training import create_run
from genai_utils.config import API_CONFIG

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

def load_model(model_path: str):
    """
    Loads the model's state dict from a file.
    
    Parameters:
    model_path (str): The path to the model file.
    """
    # Consider dynamic class typing
    model = torch.load(model_path)
    return model

def model_inference(model, image_array: np.ndarray):
    model.eval()
    with torch.no_grad():
        outputs = model(image_array)
        preds = torch.sigmoid(outputs) > 0.5
    return preds.cpu().numpy().tolist()

model_inference(load_model(download_model_from_reqistry(create_run({"goal": "download_model"}), API_CONFIG["model_name"])), np.random.rand(3, 224, 224))