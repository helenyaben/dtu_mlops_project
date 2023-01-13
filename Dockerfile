# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

RUN pip install dvc
RUN pip install "dvc[gdrive]"
RUN pip install "dvc[gs]"

# PULL DATA
ARG GDRIVE_CREDENTIALS_DATA
ENV GDRIVE_CREDENTIALS_DATA = $GDRIVE_CREDENTIALS_DATA
# RUN dvc remote modify myremote access_key_id $ENV_GDRIVE_CREDENTIALS_DATA
# RUN dvc remote modify myremote secret_access_key $ENV_GDRIVE_CREDENTIALS_DATA
WORKDIR /
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://dtumlops_project_fingers/
RUN dvc pull

ENTRYPOINT ["python", "--version"]