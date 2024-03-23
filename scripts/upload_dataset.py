from glob import glob

from tqdm import tqdm
import wandb
import yaml

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    run = wandb.init(project="dataset", job_type="dataset-creation")
    dataset = wandb.Artifact(config['upload_dataset']['dataset_name'], type="dataset")

    for path in tqdm(glob(config['upload_dataset']['data_path'])):
        dataset.add_file(path)

    run.log_artifact(dataset)
    wandb.finish()
