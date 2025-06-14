FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

USER root

RUN apt-get update && \
    apt-get install -y r-base r-cran-sf r-cran-readr r-cran-jsonlite r-cran-sp && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

USER vscode
