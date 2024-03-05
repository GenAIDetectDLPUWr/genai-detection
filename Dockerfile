FROM python:3.10-bookworm

RUN apt-get -y update && apt-get -y install git

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt