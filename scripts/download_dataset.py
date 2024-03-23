import os
import shutil
import tarfile

import wandb
import yaml

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Download from w&b
    run = wandb.init()
    artifact_id = config['download_dataset']['artifact_id']
    artifact = run.use_artifact(f'genai-detectio/dataset/{artifact_id}', type='dataset')
    artifact_dir = artifact.download()
    wandb.finish()

    # extract
    dir_name = f'artifacts/{artifact_id}'
    os.makedirs(os.path.join(dir_name, "0"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "1"), exist_ok=True)

    for filename in os.listdir(dir_name):
        if filename.endswith(".tar.gz"):
            dest_folder = "0" if filename.startswith("part-0") else "1"

            with tarfile.open(os.path.join(dir_name, filename), "r:gz") as tar:
                tar.extractall(os.path.join(dir_name, dest_folder))

            os.remove(os.path.join(dir_name, filename))

    # move to data folder
    data_raw_dir = "../data/01_raw"
    for folder_name in ["0", "1"]:
        source_folder = os.path.join(dir_name, folder_name)
        if os.path.isdir(source_folder):
            shutil.move(source_folder, os.path.join(data_raw_dir, folder_name))

    # clean up
    shutil.rmtree('artifacts/')
