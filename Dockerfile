FROM python:3.11-slim

WORKDIR /app

COPY ./preprocessing.py /app
COPY ./src /app/src
COPY ./requirements.txt /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/preprocessing.py"]