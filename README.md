# gen-ai-detection
## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [API](#api)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

## Overview
This is GenAI Image Classification project built with Kedro library.

We are trying to discriminate between images that are taken in the real world or created by hand and images generated using generative models like DALLE, Midjourney, or Stable Diffusion.

From our world:
![Real world puppies](imgs/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L25zODIzMC1pbWFnZS5qcGc.webp)

AI generated:
![Puppies generated by Stable Diffusion](imgs/4Q91mZx8dyJfTkuBOokf--4--ojjjo.jpg)

## Installation
Docker is required to run the project. To build the Docker image:
Clone the repo and run the following command:

```
docker build -t deep-learning-project .
```
or just run:
```
docker pull kabanosk/deep-learning-project
```

## Usage
To run the training and evaluation pipeline, you need to run the Docker container with the following command

```
# training
docker run -v ./data:/data --env WANDB_API_KEY=<WANDB_API_KEY> deep-learning-project --pipeline training

# evaluation
docker run -v ./data:/data --env WANDB_API_KEY=<WANDB_API_KEY> deep-learning-project --pipeline evaluation
```

### Kedro 

If you want to run Kedro manualy, you can run:

```
docker run -v ./data:/data --expose 4242 -it --entrypoint /bin/bash deep-learning-project
```

Then you can run Kedro commands:

```
kedro run
```

To visualize project run: 

```
kedro viz run --port 4242
```

Then you can open browser and go to `http://localhost:4242/`

## API
To run API see: [API](src/api/README.md)

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or concerns, please contact us at [email](mailto:your-email@example.com).
