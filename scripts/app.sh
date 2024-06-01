#!/bin/bash

uvicorn api.api:app &

python -m frontend.frontend