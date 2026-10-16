# Vaani — Indic Text to Speech

A minimal, animated web studio for **[AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS)**:
type text in one of 13 Indian languages (plus English and Hinglish), pick a male or female
voice, and listen to or download the generated speech.

- **Frontend** (`web/`): dependency-free HTML/CSS/JS. Live canvas waveforms driven by the
  Web Audio `AnalyserNode`, a click-to-seek player, staggered scroll reveals, a scrolling
  script marquee, light/dark themes and `prefers-reduced-motion` support.
- **Backend** (`server/app.py`): FastAPI wrapper around the upstream
  `TextToSpeechEngine` (FastPitch + HiFi-GAN, text normalization, transliteration, denoising).
  It loads every language whose checkpoints it finds and serves the site.

## Run it

```bash
sudo apt-get install libsndfile1-dev ffmpeg enchant   # system deps from the upstream README
./scripts/setup.sh hi ta bn                            # clone Indic-TTS + TTS fork, download models
uvicorn server.app:api --port 8000                     # http://localhost:8000
```

Leave out the language codes to download all of them (`as bn brx en en+hi gu hi kn ml mni mr or pa raj ta te`).
Each language is a few hundred MB; a GPU is used automatically when available.

Or with Docker:

```bash
docker build --build-arg LANGS="hi ta" -t vaani .
docker run -p 8000:8000 vaani            # add --gpus all for CUDA
```

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHECKPOINTS_DIR` | `./checkpoints` | Where `<lang>/fastpitch` and `<lang>/hifigan` live |
| `INDIC_TTS_DIR` | `./vendor/Indic-TTS/inference` | Upstream inference code (`src/`) |
| `LANGS` | all found | Comma-separated allow-list, e.g. `hi,ta` |
| `USE_CUDA` | `auto` | `true` / `false` to force |
| `MAX_CHARS` | `2000` | Per-request text limit |

To host the static site separately from the API, set `window.VAANI_API_BASE` in `web/config.js`.

## API

| Method | Path | Body / response |
| --- | --- | --- |
| `GET` | `/api/health` | `{ ready, languages: ["hi", …], error }` |
| `GET` | `/api/languages` | `[{ code, name, native, voices, available }]` |
| `POST` | `/api/tts` | `{ "text": "नमस्ते", "lang": "hi", "voice": "female" }` → `audio/wav` |

## Design

The UI was designed on a Claude Design canvas (website, mobile and UI-kit / motion
artboards): https://claude.ai/artifact/GteUyhxDuB2g9c97cFzBkX

## Credits

Models and inference code: AI4Bharat, IIT Madras —
*Towards Building Text-To-Speech Systems for the Next Billion Users* (ICASSP 2023), MIT licence.
