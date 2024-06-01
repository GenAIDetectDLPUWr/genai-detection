import requests
import gradio as gr
import base64
import io

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

demo.launch()