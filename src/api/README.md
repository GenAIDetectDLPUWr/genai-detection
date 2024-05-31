# FastAPI Inference API

This is a simple FastAPI application with a single inference endpoint. The application is containerized using Docker.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [API Endpoints](#api-endpoints)

## Installation

First, you need to install Docker. The installation process varies depending on your operating system. You can find the detailed instructions on the [official Docker documentation](https://docs.docker.com/get-docker/).

Once Docker is installed, navigate to the directory containing your FastAPI application and Dockerfile. You can then build your Docker image using the following command:

```bash
docker build -f environments/Dockerfile.api -t my-fastapi-app .
```

Replace `my-fastapi-app` with the name you want for your Docker image.

## Usage

After the Docker image is built, you can run your FastAPI application in a Docker container using the following command:

```bash
docker run -p 80:80 my-fastapi-app
```

This command will start a Docker container with your FastAPI application, and the application will be available at `http://localhost/inference`.

Remember to replace `my-fastapi-app` with the name you used for your Docker image.

To use the inference endpoint, you can send a POST request with an image file. Here's an example using curl:

```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost/inference
```

## API Endpoints

The application has the following endpoint:

- `POST /inference`: Accepts a JSON body and returns it back in the response.

