# Utils
## training.py

The `training.py` file contains utility functions for our project. 
### Functions
#### `create_run(project: str, group: str, name: str, config: dict) -> Run`
This function creates a new run in the Weights & Biases service. It takes the project name, group name, run name, and a dictionary of configuration parameters as inputs. It returns a `wandb.Run` instance that can be used to log metrics, hyperparameters, and artifacts to the run.

#### `log_metrics(run: Run, metrics: dict, step: int, prefix: str = "")`
This function logs the metrics to the Weights & Biases service. It takes a `wandb.Run` instance, a dictionary of metrics, a step number, and an optional prefix string. It then logs each metric with the prefix prepended to the metric name. The `step` parameter is used to specify the current step or epoch in the training loop.

#### `save_checkpoint(run: Run, model: Module, model_name: str)`
This function saves the model's state dictionary to a file and logs the file as an artifact to the Weights & Biases service. It takes a `wandb.Run` instance, a PyTorch model, and a model name as inputs. The model's state dictionary is saved to a file with the specified name.

#### `load_checkpoint(model: Module, model_path: str)`
This function loads a model's state dictionary from a file. It takes a PyTorch model and the path of the model file as inputs. The model's parameters are updated based on the loaded state dictionary.

#### `evaluate_model(model: Module, dataloader: DataLoader, criterion: Module) -> float`
This function evaluates the model on a given dataloader. It takes a PyTorch model, a data loader, and a loss criterion as inputs. It returns the total loss.

### Usage

To use the functions in this file, import the file and call the desired function. For example:

```python
from utils.training import create_run, evaluate_model, load_checkpoint, log_metrics, save_checkpoint

# Create a run 
run = create_run(
    project="my_project", 
    group="my_experiment", 
    name="my_run", 
    config={
        "learning_rate": 0.001, 
        "num_epochs": 10
    })

# Load a checkpoint
model = MyModel()
load_checkpoint(model, "path/to/model.pt")

for epoch in range(run.config["num_epochs"]):
    for i, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Log some metrics
        metrics = {"loss": loss.item()}
        log_metrics(run, metrics, step=i, prefix="train_")
        
    # Evaluate on validation set and log validation metrics
    validation_loss = evaluate_model(model, validation_loader, criterion)
    log_metrics(run, {"loss": validation_loss}, step=epoch, prefix="val_")

    # Save the model checkpoint
    save_checkpoint(run, model, model_name=f"model_epoch_{epoch}")

run.finish()
```