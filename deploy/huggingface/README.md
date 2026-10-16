---
title: Vaani Indic TTS API
emoji: 🗣️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

Speech API for Vaani, built on AI4Bharat Indic-TTS (FastPitch + HiFi-GAN).

- `GET /api/health`
- `GET /api/languages`
- `POST /api/tts` with `{"text": "नमस्ते", "lang": "hi", "voice": "female"}` → `audio/wav`
