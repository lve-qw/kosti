#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
models=${KOSTI_MODELS_DIR:-"$root/models"}
if [ ! -f "$models/manifest.json" ] || [ ! -f "$models/weights.pth" ]; then
  echo "Quality model not found in $models" >&2
  echo "Run this script from the release archive or set KOSTI_MODELS_DIR." >&2
  exit 2
fi
docker run --rm -p "${KOSTI_PORT:-8000}:8000" \
  -v "$models:/models:ro" \
  -e KOSTI_MODEL_MANIFEST="/models/manifest.json" \
  "${KOSTI_IMAGE:-kosti:0.1.0}"
