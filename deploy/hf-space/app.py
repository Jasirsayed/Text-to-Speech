"""Vaani speech API as a Gradio Space on ZeroGPU.

On first start it clones the AI4Bharat Indic-TTS inference code and downloads
the checkpoints for LANGS. The website calls the `tts` and `health` endpoints
through Gradio's HTTP API.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import spaces  # must be imported before torch on ZeroGPU
import gradio as gr
import numpy as np
from scipy.io.wavfile import write as wav_write

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "Indic-TTS"
CKPT = ROOT / "checkpoints"
RELEASE = os.getenv(
    "RELEASE_URL", "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release"
)
LANGS = [code.strip() for code in os.getenv("LANGS", "hi,ta,bn").split(",") if code.strip()]
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000"))
FEMALE_ONLY = {"brx"}


def ensure_code() -> None:
    if not (VENDOR / "inference" / "src").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/AI4Bharat/Indic-TTS", str(VENDOR)],
            check=True,
        )


def ensure_checkpoint(lang: str) -> None:
    if (CKPT / lang / "fastpitch" / "best_model.pth").exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{lang}.zip"
        print(f"Downloading {lang} checkpoints…", flush=True)
        urllib.request.urlretrieve(f"{RELEASE}/{lang}.zip", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(Path(tmp) / "out")
        # Archives may or may not have a top-level <lang>/ folder.
        fastpitch = next((Path(tmp) / "out").rglob("fastpitch"))
        CKPT.mkdir(exist_ok=True)
        shutil.rmtree(CKPT / lang, ignore_errors=True)
        shutil.move(str(fastpitch.parent), str(CKPT / lang))


ensure_code()
for code in LANGS:
    ensure_checkpoint(code)

sys.path.insert(0, str(VENDOR / "inference"))
from TTS.utils.synthesizer import Synthesizer  # noqa: E402
from src.inference import TextToSpeechEngine  # noqa: E402

models = {}
for code in LANGS:
    base = CKPT / code
    # ZeroGPU: models placed on CUDA here are attached to a GPU only inside @spaces.GPU calls.
    models[code] = Synthesizer(
        tts_checkpoint=str(base / "fastpitch" / "best_model.pth"),
        tts_config_path=str(base / "fastpitch" / "config.json"),
        tts_speakers_file=str(base / "fastpitch" / "speakers.pth"),
        tts_languages_file=None,
        vocoder_checkpoint=str(base / "hifigan" / "best_model.pth"),
        vocoder_config=str(base / "hifigan" / "config.json"),
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=True,
    )
engine = TextToSpeechEngine(models)
print("Ready:", ", ".join(LANGS), flush=True)


@spaces.GPU(duration=60)
def _infer(text: str, lang: str, voice: str):
    return engine.infer_from_text(text, lang, voice)


def tts(text: str, lang: str, voice: str) -> str:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Text is empty.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text is longer than {MAX_CHARS} characters.")
    if lang not in models:
        raise gr.Error(f"Language '{lang}' is not installed on this Space.")
    if voice not in ("female", "male") or (voice == "male" and lang in FEMALE_ONLY):
        raise gr.Error(f"The {voice} voice is not available for this language.")
    wav = _infer(text, lang, voice)
    if wav is None or len(wav) == 0:
        raise gr.Error("Nothing to speak in this text.")
    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_write(out.name, engine.target_sr, pcm)
    return out.name


def health() -> dict:
    return {"ready": True, "languages": LANGS, "error": None}


with gr.Blocks(title="Vaani Indic TTS API") as demo:
    gr.Markdown("# Vaani — Indic TTS API\nAI4Bharat FastPitch + HiFi-GAN. Used by the Vaani website.")
    with gr.Row():
        text = gr.Textbox(label="Text", value="नमस्ते, आप कैसे हैं?", lines=3)
        with gr.Column():
            lang = gr.Dropdown(LANGS, value=LANGS[0], label="Language")
            voice = gr.Radio(["female", "male"], value="female", label="Voice")
            go = gr.Button("Generate", variant="primary")
    audio = gr.Audio(type="filepath", label="Output")
    status = gr.JSON(visible=False)
    go.click(tts, [text, lang, voice], audio, api_name="tts")
    gr.Button("Health", visible=False).click(health, None, status, api_name="health")

demo.queue(default_concurrency_limit=2).launch()
