import requests
import gradio as gr
import base64
import io

import threading
import uvicorn

def encode_image(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')  # Use the appropriate image format
    encoded_img = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return encoded_img

def classify_image(image):
    response = requests.post("http://localhost:8000/inference", files={"file": encode_image(image)})
    classified_image = {1: 'AI generated', 0: 'Real image'}.get(response.json()["prediction"], "Unknown")
    return classified_image

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Classification"),
)

def run_frontend():
    demo.launch()

frontend_service = threading.Thread(target=run_frontend)
frontend_service.start()

def run_api():
    uvicorn.run("api.api:app", host="127.0.0.1", port=8000)

api_service = threading.Thread(target=run_api)
api_service.start()