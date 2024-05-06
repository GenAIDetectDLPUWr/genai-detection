FROM python:3.9-slim-buster

WORKDIR /app
COPY . /app

# TODO: Install dependencies with potry (add uvicorn, fastapi)

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]