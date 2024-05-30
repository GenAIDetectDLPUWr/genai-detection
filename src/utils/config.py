import yaml

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

CONFIG_VERSION = config["version"]

TRAIN_CONFIG = config["train_config"]

WANDB_CONFIG = config["wandb_config"]

EVALUATION_CONFIG = config["evaluation_config"]

API_CONFIG = config["api_config"]