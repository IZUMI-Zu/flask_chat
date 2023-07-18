FROM python:3.10-bullseye

WORKDIR /app

ENV VENV_NAME=flask_venv TAG=v0.2.0 MODEL_FILE_NAEME=model.tar.gz

RUN apt-get update && apt-get install -y \
    tar \
    wget \
    gunicorn \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN /app/bin/install && . /app/flask_venv/bin/activate

CMD [ "/app/bin/start" ]
