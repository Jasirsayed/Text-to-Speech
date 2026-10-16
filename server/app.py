"""Vaani — a web front-end and REST API around AI4Bharat Indic-TTS.

The Indic-TTS inference engine (``inference/src`` in the upstream repo) is
imported from ``INDIC_TTS_DIR``. Every language whose checkpoints are found
under ``CHECKPOINTS_DIR`` is loaded at startup; the website itself is served
even when no model could be loaded, and ``/api/health`` says why.
"""

import io
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
INDIC_TTS_DIR = Path(os.getenv("INDIC_TTS_DIR", ROOT / "vendor" / "Indic-TTS" / "inference"))
TTS_DIR = Path(os.getenv("TTS_DIR", ROOT / "vendor" / "TTS"))
CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", ROOT / "checkpoints"))
USE_CUDA = os.getenv("USE_CUDA", "auto")
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000"))
# Optional comma-separated allow-list, e.g. "hi,ta,bn", to limit memory use.
ONLY_LANGS = {code.strip() for code in os.getenv("LANGS", "").split(",") if code.strip()}

log = logging.getLogger("vaani")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

LANGUAGES = {
    "as": {"name": "Assamese", "native": "অসমীয়া", "voices": ["female", "male"]},
    "bn": {"name": "Bengali", "native": "বাংলা", "voices": ["female", "male"]},
    "brx": {"name": "Bodo", "native": "बड़ो", "voices": ["female"]},
    "en": {"name": "English", "native": "English", "voices": ["female", "male"]},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "voices": ["female", "male"]},
    "hi": {"name": "Hindi", "native": "हिन्दी", "voices": ["female", "male"]},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "voices": ["female", "male"]},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "voices": ["female", "male"]},
    "mni": {"name": "Manipuri", "native": "মৈতৈলোন্", "voices": ["female", "male"]},
    "mr": {"name": "Marathi", "native": "मराठी", "voices": ["female", "male"]},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ", "voices": ["female", "male"]},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "voices": ["female", "male"]},
    "raj": {"name": "Rajasthani", "native": "राजस्थानी", "voices": ["female", "male"]},
    "ta": {"name": "Tamil", "native": "தமிழ்", "voices": ["female", "male"]},
    "te": {"name": "Telugu", "native": "తెలుగు", "voices": ["female", "male"]},
}


class EngineState:
    engine = None
    loaded: list = []
    error: str | None = None
    lock = threading.Lock()


state = EngineState()


def checkpoint_paths(lang: str) -> dict:
    base = CHECKPOINTS_DIR / lang
    return {
        "tts_checkpoint": base / "fastpitch" / "best_model.pth",
        "tts_config_path": base / "fastpitch" / "config.json",
        "tts_speakers_file": base / "fastpitch" / "speakers.pth",
        "vocoder_checkpoint": base / "hifigan" / "best_model.pth",
        "vocoder_config": base / "hifigan" / "config.json",
    }


def find_installed_languages() -> list:
    found = []
    for lang in LANGUAGES:
        if ONLY_LANGS and lang not in ONLY_LANGS:
            continue
        paths = checkpoint_paths(lang)
        if paths["tts_checkpoint"].exists() and paths["vocoder_checkpoint"].exists():
            found.append(lang)
    return found


def load_engine() -> None:
    langs = find_installed_languages()
    if not langs:
        state.error = f"No checkpoints found in {CHECKPOINTS_DIR}. Run scripts/setup.sh."
        log.warning(state.error)
        return
    try:
        from server.compat import fix_checkpoint_config, install_compat_shims

        install_compat_shims()
        sys.path.insert(0, str(TTS_DIR))
        sys.path.insert(0, str(INDIC_TTS_DIR))
        from TTS.utils.synthesizer import Synthesizer
        from src.inference import TextToSpeechEngine
    except ImportError as exc:
        state.error = f"Indic-TTS is not installed ({exc}). Run scripts/setup.sh."
        log.warning(state.error)
        return

    if USE_CUDA == "auto":
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False
    else:
        use_cuda = USE_CUDA.lower() in {"1", "true", "yes"}

    models = {}
    for lang in langs:
        fix_checkpoint_config(CHECKPOINTS_DIR / lang)
        paths = {key: str(path) for key, path in checkpoint_paths(lang).items()}
        log.info("Loading %s (cuda=%s)", lang, use_cuda)
        models[lang] = Synthesizer(
            **paths,
            tts_languages_file=None,
            encoder_checkpoint="",
            encoder_config="",
            use_cuda=use_cuda,
        )
    # Transliteration (IndicXlit) and the denoiser (asteroid) pull in TensorFlow / a second
    # torchaudio, so they are off: Roman-script words are spoken as typed, audio is 22.05 kHz.
    state.engine = TextToSpeechEngine(models, allow_transliteration=False, enable_denoiser=False)
    state.loaded = langs
    state.error = None
    log.info("Ready: %s", ", ".join(langs))


api = FastAPI(title="Vaani — Indic TTS")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@api.on_event("startup")
def startup() -> None:
    # Load in the background so the website is reachable while models warm up.
    threading.Thread(target=load_engine, daemon=True).start()


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str
    voice: str = "female"


@api.get("/api/health")
def health():
    return {
        "ready": state.engine is not None,
        "languages": state.loaded,
        "error": state.error,
    }


@api.get("/api/languages")
def languages():
    return [
        {"code": code, **meta, "available": code in state.loaded}
        for code, meta in LANGUAGES.items()
    ]


@api.post("/api/tts")
def synthesize(req: SynthesisRequest):
    from scipy.io.wavfile import write as wav_write

    if state.engine is None:
        raise HTTPException(503, state.error or "Models are still loading.")
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Text is empty.")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"Text is longer than {MAX_CHARS} characters.")
    if req.lang not in state.loaded:
        raise HTTPException(404, f"Language '{req.lang}' is not installed on this server.")
    if req.voice not in LANGUAGES[req.lang]["voices"]:
        raise HTTPException(422, f"The {req.voice} voice is not available for this language.")

    with state.lock:
        wav = state.engine.infer_from_text(text, req.lang, req.voice)
        sample_rate = state.engine.target_sr

    if wav is None or len(wav) == 0:
        raise HTTPException(422, "Nothing to speak in this text.")
    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_write(buf, sample_rate, pcm)
    return Response(buf.getvalue(), media_type="audio/wav")


@api.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


api.mount("/", StaticFiles(directory=WEB_DIR), name="web")

if __name__ == "__main__":
    import uvicorn

    sys.path.insert(0, str(ROOT))
    uvicorn.run("server.app:api", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
