# Hugging Face Space (Docker SDK). Free CPU tier: 2 vCPU / 16 GB RAM.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl unzip build-essential libsndfile1-dev ffmpeg libenchant-2-2 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
COPY --chown=user . .
USER user

# Languages to bake in. Each adds a few hundred MB and ~1 GB of RAM at runtime;
# 3–4 languages fit comfortably on the free tier.
ARG LANGS="hi ta bn"
RUN ./scripts/setup.sh $LANGS

ENV PORT=7860 USE_CUDA=auto
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:api --host 0.0.0.0 --port $PORT"]
