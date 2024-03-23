from glob import glob

from tqdm import tqdm
import wandb


if __name__ == "__main__":
    run = wandb.init(project="dataset", job_type="dataset-creation")

    dataset = wandb.Artifact("stable-diffusion-laion", type="dataset")

    for path in tqdm(glob('../data/01_raw/*tar.gz')):
        dataset.add_file(path)

    run.log_artifact(dataset)
    wandb.finish()
