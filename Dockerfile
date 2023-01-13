# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# PULL DATA
ARG GDRIVE_CREDENTIALS_DATA
ENV GDRIVE_CREDENTIALS_DATA = GDRIVE_CREDENTIALS_DATA
RUN dvc remote modify myremote access_key_id ${GDRIVE_CREDENTIALS_DATA}
RUN dvc remote modify myremote secret_access_key ${GDRIVE_CREDENTIALS_DATA}
RUN dvc pull



ENTRYPOINT ["python", "--version"]