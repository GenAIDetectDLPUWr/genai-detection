from src.utils.model import load_model

import gradio as gr
import torch


def main(image):
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    model = load_model('model1')
    return f"This image is ai generated with {torch.sigmoid(model.forward(image)).item() * 100}% probability"


demo = gr.Interface(fn=main, inputs="image", outputs="text").launch()
demo.launch(share=True)
