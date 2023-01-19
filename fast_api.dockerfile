FROM python:3.9-slim

EXPOSE $PORT

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py setup.py
COPY app/ app/
COPY src/ src/
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD exec uvicorn app.main:app --port $PORT --host 0.0.0.0 --workers 1