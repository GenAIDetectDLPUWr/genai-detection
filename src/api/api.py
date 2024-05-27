from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

from utils.model import download_model_from_reqistry, load_model
from utils.training import create_run
from utils.config import API_CONFIG

app = FastAPI()

model_run = create_run({"goal": "download_model"})
model_path = download_model_from_reqistry(model_run, API_CONFIG["model_name"])
model = load_model(model_path)

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_array = np.array(image)

    raise NotImplementedError("You need to add inference pipeline here!")
