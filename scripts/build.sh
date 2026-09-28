#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker build --build-arg INSTALL_ML="${KOSTI_INSTALL_ML:-1}" -t "${KOSTI_IMAGE:-kosti:0.1.0}" .
