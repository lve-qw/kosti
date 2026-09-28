#!/bin/bash
# Run on the deployment host: bash promote-model.sh /home/masha/kosti-deploy/models-VERSION
# Validate a staged model with the current image before replacing the live model.
set -euo pipefail
staged=${1:?Staged model directory required}
[[ "$staged" =~ ^/home/masha/kosti-deploy/models-[a-zA-Z0-9-]+$ ]] || exit 2
test -d "$staged"
test ! -L "$staged"
test -f "$staged/manifest.json"
test -f "$staged/weights.pth"
exec 9>/home/masha/kosti-deploy.lock
flock -w 600 9
image=$(docker inspect kosti --format '{{.Config.Image}}')
models=/home/masha/kosti-deploy/models
stamp=$(date +%Y%m%d%H%M%S)
backup="/home/masha/kosti-deploy/models-backup-$stamp"
previous="kosti-before-model-$stamp"
candidate="kosti-model-check-$stamp"
cleanup() { docker rm -f "$candidate" >/dev/null 2>&1 || true; }
trap cleanup EXIT
run() {
  docker run -d --name "$1" --restart unless-stopped --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g --pids-limit 256 --memory 2g \
    --security-opt no-new-privileges --cap-drop ALL \
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    -p "$2:8000" -v "$3:/models:ro" \
    -e KOSTI_MODEL_MANIFEST=/models/manifest.json "$image"
}
ready() {
  for attempt in $(seq 1 45); do
    if curl -fsS --max-time 3 "http://127.0.0.1:$1/ready" >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}
run "$candidate" 127.0.0.1:18000 "$staged"
ready 18000
# Synthetic pixels only: checks actual HTTP inference without uploading patient data.
docker exec -i "$candidate" python - <<'PY'
import csv, io, json, time, urllib.request, zipfile
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
meta = FileMetaDataset()
meta.TransferSyntaxUID = ExplicitVRLittleEndian
meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
meta.MediaStorageSOPInstanceUID = generate_uid()
ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
ds.SOPClassUID = meta.MediaStorageSOPClassUID
ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()
ds.Rows, ds.Columns = 256, 256
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = 'MONOCHROME2'
ds.BitsAllocated = ds.BitsStored = 16
ds.HighBit, ds.PixelRepresentation = 15, 0
ds.PixelData = np.arange(65536, dtype=np.uint16).reshape(256,256).tobytes()
stream = io.BytesIO()
ds.save_as(stream, enforce_file_format=True)
boundary = 'kosti-model-smoke-boundary'
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="synthetic.dcm"\r\nContent-Type: application/dicom\r\n\r\n'.encode()
        + stream.getvalue() + f'\r\n--{boundary}--\r\n'.encode())
request = urllib.request.Request('http://127.0.0.1:8000/v1/batch', data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
started = time.monotonic()
with urllib.request.urlopen(request, timeout=180) as response:
    with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read('results.csv').decode('utf-8-sig'))))
assert len(rows) == 1 and rows[0]['processing_status'] == 'Success', rows
assert time.monotonic() - started < 180
print(json.dumps({'synthetic_http_inference': 'passed', 'seconds': time.monotonic()-started}))
PY
cleanup
test ! -e "$backup"
docker stop kosti
docker rename kosti "$previous"
if ! mv "$models" "$backup"; then
  docker rename "$previous" kosti
  docker start kosti
  exit 1
fi
if ! mv "$staged" "$models"; then
  mv "$backup" "$models"
  docker rename "$previous" kosti
  docker start kosti
  exit 1
fi
if run kosti 8000 "$models" && ready 8000; then
  docker rm "$previous"
  echo "MODEL_PROMOTED backup=$backup image=$image"
else
  docker logs kosti || true
  docker rm -f kosti || true
  mv "$models" "$staged"
  mv "$backup" "$models"
  docker rename "$previous" kosti
  docker start kosti
  echo 'Model promotion failed; previous model and container restored' >&2
  exit 1
fi
