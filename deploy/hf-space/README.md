---
title: Vaani Indic TTS API
emoji: 🗣️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

Speech API for Vaani, built on AI4Bharat Indic-TTS (FastPitch + HiFi-GAN), running on ZeroGPU.

API endpoints (Gradio): `tts(text, lang, voice)` → WAV file, `health()` → status JSON.
Set the `LANGS` variable in Space settings (default `hi,ta,bn`) to choose languages.
