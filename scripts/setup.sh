#!/usr/bin/env bash
# Installs the Indic-TTS inference stack and downloads model checkpoints.
#
#   scripts/setup.sh            # all languages
#   scripts/setup.sh hi ta bn   # only these
#
# Needs: python3.10+, git, curl, unzip, and system packages
#   libsndfile1-dev ffmpeg enchant (apt) — see the upstream README.
set -euo pipefail

cd "$(dirname "$0")/.."
RELEASE_URL="${RELEASE_URL:-https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release}"
ALL_LANGS="as bn brx en gu hi kn ml mni mr or pa raj ta te"
LANGS="${*:-$ALL_LANGS}"

mkdir -p vendor checkpoints

if [ ! -d vendor/Indic-TTS ]; then
  git clone --depth 1 https://github.com/AI4Bharat/Indic-TTS vendor/Indic-TTS
fi
if [ ! -d vendor/TTS ]; then
  # AI4Bharat's checkpoints need this Coqui TTS fork.
  git clone --depth 1 https://github.com/gokulkarthik/TTS vendor/TTS
fi

# The TTS fork pins 2022-era packages, so it is used from vendor/TTS via sys.path
# (see server/compat.py) and only its runtime deps are installed, at current versions.
python3 -m pip install torch
python3 -m pip install -r deploy/hf-space/requirements.txt -r server/requirements.txt

for lang in $LANGS; do
  if [ -f "checkpoints/$lang/fastpitch/best_model.pth" ]; then
    echo "✓ $lang already downloaded"
    continue
  fi
  echo "↓ $lang"
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/$lang.zip" "$RELEASE_URL/$lang.zip"
  unzip -q "$tmp/$lang.zip" -d "$tmp/out"
  # Archives may or may not have a top-level <lang>/ folder; locate fastpitch/.
  src="$(dirname "$(find "$tmp/out" -type d -name fastpitch | head -n1)")"
  rm -rf "checkpoints/$lang" && mv "$src" "checkpoints/$lang"
  rm -rf "$tmp"
done

echo "Done. Start the site with: uvicorn server.app:api --port 8000"
