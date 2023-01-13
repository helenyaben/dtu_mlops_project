# Base image
FROM python:3.7-slim

# Install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

WORKDIR /

RUN pip install -r requirements.txt

# Pull data from bucket
ARG GDRIVE_CREDENTIALS_DATA
ENV GDRIVE_CREDENTIALS_DATA = $GDRIVE_CREDENTIALS_DATA
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://dtumlops_project_fingers/
RUN dvc pull

# Set Wandb api environmental variable
ARG WANDB_API_KEY
ENV WANDB_API_KEY = $WANDB_API_KEY

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]