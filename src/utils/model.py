"""Utilities for model download and loading"""

import wandb
from wandb.sdk.wandb_run import Run
from pathlib import Path
import os
import torch
import numpy as np
from torchvision import models, transforms

from utils.training import create_run
from utils.config import API_CONFIG


def define_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 1)
    return model


def download_model_from_reqistry(model_name: str, version: str = "latest", overwrite: bool = False):
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
        run = create_run({"goal": "download_model"})
        downloaded_model_path = Path(run.use_model(name=f"{model_name}:{version}"))
        os.rename(downloaded_model_path, destination_path)
        run.finish()

    return destination_path

def load_model(model_path: str):
    """
    Loads the model's state dict from a file.
    
    Parameters:
    model_path (str): The path to the model file.
    """
    # Consider dynamic class typing
    model = define_model()
    model.load_state_dict(torch.load(model_path))
    return model

def model_inference(model, image_array: np.ndarray):
    model.eval()
    input_image = model_preprocessing(image_array)
    with torch.no_grad():
        outputs = model(input_image.unsqueeze(0))
        preds = torch.sigmoid(outputs) > 0.5
    return preds.cpu().numpy().flatten().tolist()

def model_preprocessing(image_array: torch.Tensor):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return preprocess(image_array)
