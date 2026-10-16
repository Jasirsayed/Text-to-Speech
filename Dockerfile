FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl unzip build-essential libsndfile1-dev ffmpeg libenchant-2-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY scripts ./scripts
COPY server ./server
COPY web ./web

# Pass e.g. --build-arg LANGS="hi ta bn" to bake in only some languages.
ARG LANGS=""
RUN ./scripts/setup.sh $LANGS

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server.app:api --host 0.0.0.0 --port $PORT"]
