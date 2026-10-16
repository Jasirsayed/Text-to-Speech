#!/usr/bin/env bash
# Builds dist/vaani-hf-space.zip: the files for a Hugging Face Gradio Space (ZeroGPU).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/vaani-hf-space.zip
(cd deploy/hf-space && zip -q ../../dist/vaani-hf-space.zip app.py requirements.txt packages.txt README.md)
echo "Built dist/vaani-hf-space.zip"
