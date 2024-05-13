# FastAPI Inference API

This is a simple FastAPI application with a single inference endpoint. The application is containerized using Docker.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [API Endpoints](#api-endpoints)

## Installation

Complete the installation process from main README.md.

## Usage

Build the Docker Container
```
docker run -it -p 80:80 --name dlp deep-learning-project
```
and open it in bash 
```
docker exec -it dlp bash
```

After the Docker image is built, you can run your FastAPI application in a Docker container using the following command:

```bash
uvicorn --port 80 src.api.api:app
```

This command will start your FastAPI app, and it will be available at `http://localhost/inference`.

To use the inference endpoint, you can send a POST request with an image file. Here's an example using curl:

```bash
curl -F "file=@/path/to/your/image.jpg" -X POST http://localhost/inference
```

## API Endpoints

The application has the following endpoint:

- `POST /inference`: Accepts a JSON body and returns it back in the response.

