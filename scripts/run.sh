#!/bin/sh
set -eu
if [ -n "${KOSTI_MODELS_DIR:-}" ]; then
  docker run --rm -p "${KOSTI_PORT:-8000}:8000" \
    -v "$KOSTI_MODELS_DIR:/models:ro" \
    -e KOSTI_MODEL_MANIFEST="/models/manifest.json" \
    "${KOSTI_IMAGE:-kosti:0.1.0}"
else
  docker run --rm -p "${KOSTI_PORT:-8000}:8000" "${KOSTI_IMAGE:-kosti:0.1.0}"
fi
