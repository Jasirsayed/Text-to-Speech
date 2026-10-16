"""Make the 2022-era AI4Bharat checkpoints and Coqui TTS fork run on current packages.

Kept in sync with deploy/hf-space/app.py, which must stay a single uploadable file.
"""

import json
import sys
import types
from pathlib import Path


def fix_checkpoint_config(lang_dir: Path) -> None:
    """The released configs point speakers_file at the authors' own models/v1/<lang>/ path."""
    config_path = lang_dir / "fastpitch" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    speakers = str(lang_dir / "fastpitch" / "speakers.pth")
    if config.get("speakers_file") != speakers:
        config["speakers_file"] = speakers
        config.setdefault("model_args", {})["speakers_file"] = speakers
        config_path.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8")


def install_compat_shims() -> None:
    def stub(name: str, **attrs) -> None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module

    # torchaudio must match torch exactly; FastPitch inference never calls it.
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
