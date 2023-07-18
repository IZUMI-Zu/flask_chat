FROM python:3.10-bullseye

WORKDIR /app

ENV VENV_NAME=flask_venv TAG=v0.2.0 MODEL_FILE_NAEME=model.tar.gz

RUN apt-get update && apt-get install -y \
    tar \
    wget \
    gunicorn \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/IZUMI-Zu/flask_chat.git . && rm -rf .git

RUN /app/bin/install && . /app/${VENV_NAME}/bin/activate

CMD [ "sh", "-c", ". /app/${VENV_NAME}/bin/activate && /app/bin/start" ]
