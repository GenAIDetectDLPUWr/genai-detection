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
    artifact_id = config['test_download']['artifact_id']
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
    data_test_dir = "tests/data/"
    for folder_name in ["0", "1"]:
        source_folder = os.path.join(dir_name, folder_name)
        if os.path.isdir(source_folder):
            shutil.move(source_folder, os.path.join(data_test_dir, folder_name))

    shutil.rmtree('artifacts')

    # Check if all files are downloaded
    paths = [
        '0/000000000.jpg',
        '0/000000001.jpg',
        '0/000000002.jpg',
        '0/000000003.jpg',
        '0/000000004.jpg',
        '0/000000005.jpg',
        '0/000000006.jpg',
        '0/000000007.jpg',
        '0/000000008.jpg',
        '0/000000009.jpg',
        '1/00009140-d735-4006-9da1-2b3e5a800df9.png',
        '1/00009c63-5115-4f81-b7ff-83a61e98b119.png',
        '1/00015464-a692-42ab-9eff-808d5e852360.png',
        '1/00019cc7-af4a-492e-b494-5aa7b851abb0.png',
        '1/0001e5e8-3c99-490a-b59f-e41c8a6627ce.png',
        '1/00020510-f831-4b8b-8bd7-562ff05ee213.png',
        '1/00029270-9aa5-458c-b00c-b34b66beadf9.png',
        '1/00232cb7-863c-451a-bba6-703c0d84916e.png',
        '1/00235489-f506-49b1-9414-4ceae119ce6a.png',
        '1/0023d52b-e9f3-458f-a83c-e1c73a98e9b5.png'
    ]

    for path in paths:
        assert os.path.exists(os.path.join(data_test_dir, path)), f"Path {data_test_dir}{path} does not exist"
        print(f"Path {path} exists")
