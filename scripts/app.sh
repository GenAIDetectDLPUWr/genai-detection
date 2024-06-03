#!/bin/bash
wandb login

uvicorn api.api:app &

python -m frontend.frontend