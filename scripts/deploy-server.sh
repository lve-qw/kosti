#!/bin/bash
set -euo pipefail
commit=${SSH_ORIGINAL_COMMAND:-${1:-}}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected commit SHA'; exit 2; }
exec 9>/home/masha/kosti-deploy.lock
flock -w 600 9
mkdir -p /home/masha/kosti-releases
release=$(mktemp -d "/home/masha/kosti-releases/${commit}.XXXXXX")
tar -xzf - -C "$release" --no-same-owner
cd "$release"
docker build --build-arg INSTALL_ML=1 -t "kosti:$commit" .
models=/home/masha/kosti-deploy/models
test -f "$models/manifest.json"
candidate="kosti-check-${commit:0:12}"
cleanup() { docker rm -f "$candidate" >/dev/null 2>&1 || true; }
trap cleanup EXIT
run() {
  docker run -d --name "$1" --restart unless-stopped --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g --pids-limit 256 --memory 2g \
    --security-opt no-new-privileges --cap-drop ALL \
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -p "$2:8000" -v "$models:/models:ro" \
    -e KOSTI_MODEL_MANIFEST=/models/manifest.json "kosti:$commit"
}
ready() {
  for attempt in $(seq 1 45); do
    if curl -fsS --max-time 3 "http://127.0.0.1:$1/ready" >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}
run "$candidate" 127.0.0.1:18000
ready 18000 || { docker logs "$candidate"; exit 1; }
cleanup
previous="kosti-previous-$(date +%s)"
had_previous=false
if docker inspect kosti >/dev/null 2>&1; then
  docker stop kosti
  docker rename kosti "$previous"
  had_previous=true
fi
if run kosti 8000 && ready 8000; then
  if "$had_previous"; then docker rm "$previous"; fi
  echo "DEPLOYED $commit"
else
  docker logs kosti || true
  docker rm -f kosti || true
  if "$had_previous"; then docker rename "$previous" kosti; docker start kosti; fi
  echo 'Deployment failed; previous container restored' >&2
  exit 1
fi
