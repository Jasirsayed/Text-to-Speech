#!/usr/bin/env bash
# Builds dist/vaani-hf-space.zip: everything a Hugging Face Docker Space needs.
set -euo pipefail
cd "$(dirname "$0")/.."
out=dist/hf-space
rm -rf "$out" && mkdir -p "$out"
cp -r server scripts web "$out/"
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md "$out/"
find "$out" -name __pycache__ -prune -exec rm -rf {} +
(cd "$out" && rm -f ../vaani-hf-space.zip && zip -qr ../vaani-hf-space.zip .)
echo "Built dist/vaani-hf-space.zip"
