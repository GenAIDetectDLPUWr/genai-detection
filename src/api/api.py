from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

app = FastAPI()


@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_array = np.array(image)

    raise NotImplementedError("You need to add inference pipeline here!")
