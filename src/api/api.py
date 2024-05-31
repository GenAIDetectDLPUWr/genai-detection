from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

from utils.model import download_model_from_reqistry, load_model, model_inference
from utils.config import API_CONFIG

app = FastAPI()

model_path = download_model_from_reqistry(API_CONFIG["model_name"])
model = load_model(model_path)

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_array = np.array(image)
    prediction = model_inference(model, image_array)

    return {"prediction": prediction}