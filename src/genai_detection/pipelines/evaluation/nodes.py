from kedro.pipeline import Pipeline, pipeline, node

from torchvision import transforms, models
import torch

from genai_detection.settings import EVALUATION_CONFIG
from utils.training import create_run, log_metrics
from utils.model import load_model

from pathlib import Path

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
    model_path = Path(EVALUATION_CONFIG["model_path"]) / f'{EVALUATION_CONFIG["model_name"]}.pt'
    model = load_model(model_path)
    return model


model_loading_node = node(
                func=get_model,
                inputs=None,
                outputs="evalution_model",
                name="model_loading",
)


def initialize_wandb_run():
    """Initialize a W&B run."""
    return create_run(config=EVALUATION_CONFIG)


initialize_eval_wandb_run_node = node(
                func=initialize_wandb_run,
                inputs=None,
                outputs="eval_wandb_run",
                name="wandb_eval_run_initialization",
)

def get_data_loader(dataset):
    """Get the DataLoader for the given dataset."""
    return torch.utils.data.DataLoader(dataset, batch_size=EVALUATION_CONFIG["batch_size"], shuffle=False)


get_data_loader_node = node(
                func=get_data_loader,
                inputs="preprocessed_data",
                outputs="dataloader",
                name="dataloader_creation",
)


def make_predictions(model, dataloader):
    """Evaluaten the model on the given dataset based on config yaml."""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            outputs = model(batch_inputs)
            batch_preds = torch.sigmoid(outputs) > 0.5
            preds.extend(batch_preds.cpu().numpy().tolist())
            targets.extend(batch_targets.cpu().numpy().tolist())
    return torch.Tensor(preds), torch.Tensor(targets)

make_predictions_node = node(
                func=make_predictions,
                inputs=["evalution_model", "dataloader"],
                outputs=["predictions", "targets"],
                name="inference",
)

def evalaute_predicitons(predictions, targets):
    """Evaluate the model's performance on the dataset."""
    metrics = {}
    metrics["accuracy"] = (predictions == targets).sum().item() / len(targets)
    metrics["precision"] = (predictions * targets).sum().item() / predictions.sum().item()
    metrics["recall"] = (predictions * targets).sum().item() / targets.sum().item()
        
    return metrics

evaluate_predictions_node = node(
                func=evalaute_predicitons,
                inputs=["predictions", "targets"],
                outputs="metrics",
                name="evaluation",
)

def log_eval_metrics(run, metrics):
    """Log the evaluation metrics to W&B."""
    log_metrics(run=run, metrics=metrics, step=-1)
    return run

log_metrics_node = node(
                func=log_eval_metrics,
                inputs=["eval_wandb_run", "metrics"],
                outputs="finished_eval_wandb_run",
                name="metrics_logging",
)

def finish_wandb_run(wandb_run) -> None:
    wandb_run.finish()

finish_wandb_run_node = node(
                func=finish_wandb_run,
                inputs="finished_eval_wandb_run",
                outputs=None,
                name="wandb_eval_run_finishing",
)