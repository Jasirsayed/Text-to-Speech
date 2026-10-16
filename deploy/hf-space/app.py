"""Vaani speech API as a Gradio Space on ZeroGPU.

On first start it clones the AI4Bharat Indic-TTS inference code and the Coqui
TTS fork its checkpoints need, and downloads the checkpoints for LANGS. The
website calls the `tts` and `health` endpoints through Gradio's HTTP API.

The TTS fork pins 2022-era packages that cannot be installed next to the
PyTorch ZeroGPU provides, so it is put on sys.path instead of pip-installed,
and the few modules it imports but never uses for FastPitch inference are
stubbed (see `install_compat_shims`).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.request
import zipfile
from pathlib import Path

import spaces  # must be imported before torch on ZeroGPU
import gradio as gr
import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

ROOT = Path(__file__).resolve().parent
VENDOR = Path(os.getenv("VENDOR_DIR", ROOT / "vendor"))
CKPT = Path(os.getenv("CHECKPOINTS_DIR", ROOT / "checkpoints"))
RELEASE = os.getenv(
    "RELEASE_URL", "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release"
)
LANGS = [code.strip() for code in os.getenv("LANGS", "hi,ta,bn").split(",") if code.strip()]
MAX_CHARS = int(os.getenv("MAX_CHARS", "2000"))
FEMALE_ONLY = {"brx"}
REPOS = {
    "Indic-TTS": "https://github.com/AI4Bharat/Indic-TTS",
    "TTS": "https://github.com/gokulkarthik/TTS",
}


def ensure_code() -> None:
    for name, url in REPOS.items():
        if not (VENDOR / name / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(VENDOR / name)], check=True)


def ensure_checkpoint(lang: str) -> None:
    target = CKPT / lang
    if not (target / "fastpitch" / "best_model.pth").exists():
        with tempfile.TemporaryDirectory(dir=CKPT.parent) as tmp:
            archive = Path(tmp) / f"{lang}.zip"
            print(f"Downloading {lang} checkpoints (~1.5 GB)…", flush=True)
            urllib.request.urlretrieve(f"{RELEASE}/{lang}.zip", archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(Path(tmp) / "out")
            archive.unlink()
            # Archives may or may not have a top-level <lang>/ folder.
            fastpitch = next((Path(tmp) / "out").rglob("fastpitch"))
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(fastpitch.parent), str(target))
    # The released configs point speakers_file at the authors' own models/v1/<lang>/ path.
    config_path = target / "fastpitch" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    speakers = str(target / "fastpitch" / "speakers.pth")
    if config.get("speakers_file") != speakers:
        config["speakers_file"] = speakers
        config.setdefault("model_args", {})["speakers_file"] = speakers
        config_path.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8")


def install_compat_shims() -> None:
    def stub(name: str, **attrs) -> None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module

    # torchaudio must match the preinstalled torch exactly; FastPitch inference never calls it.
    try:
        import torchaudio  # noqa: F401
    except Exception:
        stub("torchaudio", transforms=types.SimpleNamespace())
    # Japanese phonemizer builds a MeCab tagger at import time; unused for Indic languages.
    try:
        import MeCab  # noqa: F401
    except Exception:
        stub("MeCab", Tagger=lambda *args, **kwargs: None)

    # The fork calls librosa.filters.mel positionally, which librosa >= 0.10 rejects.
    import librosa

    mel = librosa.filters.mel

    def mel_compat(*args, **kwargs):
        kwargs.update(zip(("sr", "n_fft"), args))
        return mel(**kwargs)

    librosa.filters.mel = mel_compat

    # Upstream scrapes Google Translate at startup to word-ify dates and numbers in
    # languages num2words lacks; keep the text unchanged instead.
    class PassthroughTranslator:
        def __call__(self, text, **kwargs):
            return text

    stub("src.utils.translator", GoogleTranslator=PassthroughTranslator)


CKPT.mkdir(parents=True, exist_ok=True)
ensure_code()
for code in LANGS:
    ensure_checkpoint(code)

install_compat_shims()
sys.path.insert(0, str(VENDOR / "TTS"))
sys.path.insert(0, str(VENDOR / "Indic-TTS" / "inference"))
from TTS.utils.synthesizer import Synthesizer  # noqa: E402
from src.inference import TextToSpeechEngine  # noqa: E402

# On ZeroGPU, CUDA reports available here and models are attached to a GPU only inside @spaces.GPU.
USE_CUDA = torch.cuda.is_available()
models = {}
for code in LANGS:
    base = CKPT / code
    models[code] = Synthesizer(
        tts_checkpoint=str(base / "fastpitch" / "best_model.pth"),
        tts_config_path=str(base / "fastpitch" / "config.json"),
        tts_speakers_file=str(base / "fastpitch" / "speakers.pth"),
        tts_languages_file=None,
        vocoder_checkpoint=str(base / "hifigan" / "best_model.pth"),
        vocoder_config=str(base / "hifigan" / "config.json"),
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=USE_CUDA,
    )
# Transliteration (IndicXlit) and the denoiser (asteroid) pull in TensorFlow / a second
# torchaudio, so they are off: Roman-script words are spoken as typed, audio is 22.05 kHz.
engine = TextToSpeechEngine(models, allow_transliteration=False, enable_denoiser=False)
print("Ready:", ", ".join(LANGS), "| cuda:", USE_CUDA, flush=True)


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
