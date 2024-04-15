from pathlib import Path

import torch
from torchvision.models import resnet18


def load_model(model_name: str) -> resnet18:
    """
    Loads the model's state dict from a file.

    Parameters:
    model_name (str): The name of the model.
        The model will be loaded from 'models/{model_name}.pt' if exists.
        If model don't exist, it will be downloaded from wandb.
    """
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    if Path(f'src/models/{model_name}.pt').exists():
        m = torch.load(f'src/models/{model_name}.pt')
        for name, layer in m.named_children():
            print(name, layer)

        model.load_state_dict(torch.load(f'src/models/{model_name}.pt'))
    else:
        # TODO: Add downloading from wandb
        pass
    return model
