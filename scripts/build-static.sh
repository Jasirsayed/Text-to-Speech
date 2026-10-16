#!/usr/bin/env bash
# Builds dist/vaani-hostinger.zip: the static site for shared hosting (Hostinger, cPanel, etc.).
#   scripts/build-static.sh [API_BASE_URL]
set -euo pipefail
cd "$(dirname "$0")/.."
API="${1:-}"
out=dist/hostinger
rm -rf "$out" && mkdir -p "$out"
cp web/index.html web/styles.css web/app.js web/config.js deploy/hostinger/.htaccess "$out/"
if [ -n "$API" ]; then
  sed -i.bak "s#window.VAANI_API_BASE = '';#window.VAANI_API_BASE = '${API%/}';#" "$out/config.js" && rm "$out/config.js.bak"
fi
# Cache-bust: browsers keep CSS/JS for a week (see .htaccess), so version each URL by content.
for f in styles.css config.js app.js; do
  v=$(sha1sum "$out/$f" | cut -c1-8)
  sed -i.bak "s#\"$f\"#\"$f?v=$v\"#" "$out/index.html" && rm "$out/index.html.bak"
done
(cd "$out" && rm -f ../vaani-hostinger.zip && zip -qr ../vaani-hostinger.zip . -x '.DS_Store')
cp deploy/hostinger/UPLOAD.txt dist/
echo "Built dist/vaani-hostinger.zip"
