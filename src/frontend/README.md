# Gradio Frontend App

This is a Gradio frontend application that allows users to interact with the model's API through a web interface. The application is containerized using Docker.

## Running the Application

First, you need to install Docker. The installation process varies depending on your operating system. You can find the detailed instructions on the [official Docker documentation](https://docs.docker.com/get-docker/).

Once Docker is installed, navigate to the directory containing the repository. You can then build your Docker image using the following command:

```bash
docker build -f environments/Dockerfile.app -t genai-detection-app .
```

Replace `genai-detection-app` with the name you want for your Docker image.

## Usage

After the Docker image is built, you can run your Gradio application in a Docker container using the following command:

```bash
docker run -p 7860:7860 genai-detection-app
```

Now, go to your browser at `http://localhost:7860/` to interact with the Gradio application.