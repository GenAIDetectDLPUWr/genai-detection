from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import base64
import io

from utils.model import download_model_from_reqistry, load_model, model_inference
from utils.config import API_CONFIG

app = FastAPI()

model_path = download_model_from_reqistry(API_CONFIG["model_name"])
model = load_model(model_path)

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    encoded_image = await file.read()
    decoded_image = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(decoded_image))
    image_array = np.array(image)
    prediction = int(model_inference(model, image_array)[0])

    return {"prediction": prediction}