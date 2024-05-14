FROM python:3.11-slim

WORKDIR /app

COPY src/data_pipeline.py /app
COPY src/import.py /app
COPY ./src /app/src

COPY ./requirements.txt /app

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir pyarrow