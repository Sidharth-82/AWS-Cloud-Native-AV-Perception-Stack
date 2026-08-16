#!/usr/bin/env bash
#
# Full Phase 1 pipeline: capture (GPU) then offline processing (CPU), both on this
# instance. Starts the CARLA server, installs the additional maps (Town06/07) if
# missing, builds both images, captures every run, then processes each split into a
# KITTI tree and publishes ONLY that tree to S3.
#
# Raw frames never leave the box. They stay in ./_scratch until the instance is
# terminated, so a threshold that looks wrong in the sample renders can be
# re-processed without paying for the GPU capture again.
#
# Run on the EC2 box from the Phase 1 dir (the one holding src/create_data/, src/process_data/,
# src/common/, config/):
#   bash orchestrate.sh                      # subset smoke: 5 frames/scene, local only
#   FRAMES= UPLOAD=1 bash orchestrate.sh     # full run, publish to S3
#
# Env knobs:
#   FRAMES   frames per scene (default 5; FRAMES= means the whole scene)
#   UPLOAD   1 to publish the processed trees to S3 (default 0, local only)
#   RUNS     capture run ids (default: every run in metadata.json)
#   SPLITS   splits to process (default: train val test)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SERVER_IMAGE="carlasim/carla:0.9.15"
CLIENT_IMAGE="carla-client:0.9.15"
OFFLINE_IMAGE="perception-offline:1.0"

FRAMES="${FRAMES-5}"
UPLOAD="${UPLOAD:-0}"
SPLITS="${SPLITS:-train val test}"
RUNS="${RUNS:-$(python3 -c "import json; print(' '.join(str(r['run_id']) for r in json.load(open('config/metadata.json'))['runs']))")}"

SCRATCH=/workspace/_scratch      # raw capture output (container path; ./_scratch on host)
KITTI=/workspace/kitti           # processed dataset root (./kitti on host)

# 1. Start the CARLA server container (idempotent — reuse if already up).
if docker ps --format '{{.Names}}' | grep -qx carla; then
    echo "[orchestrate] CARLA server already running."
else
    echo "[orchestrate] starting CARLA server..."
    docker rm -f carla >/dev/null 2>&1 || true
    docker run -d --name carla --gpus all --net=host "$SERVER_IMAGE" \
        ./CarlaUE4.sh -RenderOffScreen -nosound
fi

# 2. Wait for the RPC port to accept connections (bash /dev/tcp, no extra tools).
echo "[orchestrate] waiting for CARLA on :2000 ..."
ready=0
for _ in $(seq 1 100); do
    if (echo > /dev/tcp/localhost/2000) 2>/dev/null; then
        ready=1
        break
    fi
    sleep 2
done
if [ "$ready" -ne 1 ]; then
    echo "[orchestrate] ERROR: CARLA never opened :2000. Server log:" >&2
    docker logs --tail 40 carla >&2 || true
    exit 1
fi
echo "[orchestrate] port 2000 open; letting the default map settle..."
sleep 30   # port opens before the world is fully ready for RPC/load_world

# 2b. Ensure the additional maps (Town06/07) are in the server container. They are NOT
#     in the base carlasim/carla image, so extract the official AdditionalMaps package
#     directly into the CARLA root (NOT ImportAssets.sh, which drops only support files).
#     Installed here at runtime rather than baked into the image: committing the writable
#     layer (~15GB across tens of thousands of tiny asset files) is pathologically slow.
#     Idempotent -- skips when Town06 already present; caches the tarball to skip re-download.
MAPS_URL="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz"
MAPS_TARBALL="${MAPS_TARBALL:-$HOME/AdditionalMaps_0.9.15.tar.gz}"
TOWN06_UMAP="/home/carla/CarlaUE4/Content/Carla/Maps/Town06.umap"
if docker exec carla test -f "$TOWN06_UMAP"; then
    echo "[orchestrate] additional maps already present (Town06 found)."
else
    echo "[orchestrate] additional maps missing; installing Town06/07..."
    if [ ! -s "$MAPS_TARBALL" ]; then
        echo "[orchestrate] downloading AdditionalMaps_0.9.15 (~6.9GB) -> $MAPS_TARBALL ..."
        wget -q --show-progress -O "$MAPS_TARBALL" "$MAPS_URL"
    fi
    echo "[orchestrate] extracting maps into the CARLA root (streamed)..."
    docker exec -i carla bash -lc 'cd /home/carla && tar -xzf -' < "$MAPS_TARBALL"
    docker exec carla test -f "$TOWN06_UMAP" \
        || { echo "[orchestrate] ERROR: Town06 still missing after extract." >&2; exit 1; }
    echo "[orchestrate] maps installed; restarting server to load them..."
    docker restart carla
    ready=0
    for _ in $(seq 1 100); do
        if (echo > /dev/tcp/localhost/2000) 2>/dev/null; then ready=1; break; fi
        sleep 2
    done
    [ "$ready" -eq 1 ] || { echo "[orchestrate] ERROR: server never reopened :2000 after maps." >&2; docker logs --tail 40 carla >&2; exit 1; }
    sleep 30
fi

# 3. Build both images. Separate on purpose: the capture client is pinned to python 3.7
#    by the carla wheel, while the offline half is plain numpy/PIL CPU work on 3.11.
#    Neither image COPYs source -- both mount $HERE at /workspace -- so editing code
#    needs no rebuild. Each build context is just its own folder (no source to send).
echo "[orchestrate] building capture image ($CLIENT_IMAGE)..."
docker build -t "$CLIENT_IMAGE" -f src/create_data/Dockerfile src/create_data

echo "[orchestrate] building offline image ($OFFLINE_IMAGE)..."
docker build -t "$OFFLINE_IMAGE" -f src/process_data/Dockerfile src/process_data

# 4. CAPTURE (GPU). One client process per run, and every run is exactly one map. That
#    dodges CARLA's cross-reload client-thread crash (PyEval_SaveThread: NULL tstate):
#    each load_world is a fresh process's first op with no lingering sensor threads, and
#    scene.py's dedup reuses the map for the rest of the run. Per-run processes also keep
#    provenance clean -- each stamps its own metadata entry.
#    --net=host so the client reaches localhost:2000 AND the IMDS role creds. Mount
#    HERE -> /workspace so config/ resolves and output lands in ./_scratch on the host.
#    Nothing goes to S3 here: capture only uploads raw when handed --upload-raw.
SUBSET=()
[ -n "$FRAMES" ] && SUBSET=(--max-frames-per-scene "$FRAMES")
echo "[orchestrate] capturing runs: $RUNS (frames/scene: ${FRAMES:-all})"
for run in $RUNS; do
    echo "[orchestrate] --- capture run $run ---"
    docker run --rm --net=host -v "$HERE":/workspace "$CLIENT_IMAGE" \
        python -u src/create_data/capture.py --run "$run" \
        --output-root "$SCRATCH" "${SUBSET[@]}"
done

# 5. PROCESS (CPU). One invocation per split. Each selects the runs flagged with that
#    split in metadata.json and writes ./kitti/<split>/. No GPU and no CARLA -- it only
#    reads ./_scratch off local disk. Runs on this same box because the instance is
#    already paid for and the raw data is already sitting here.
PUBLISH=(--no-upload)
[ "$UPLOAD" = "1" ] && PUBLISH=()
echo "[orchestrate] processing splits: $SPLITS (upload: $UPLOAD)"
for split in $SPLITS; do
    echo "[orchestrate] --- process split $split ---"
    docker run --rm --net=host -v "$HERE":/workspace "$OFFLINE_IMAGE" \
        python -u src/process_data/process.py --split "$split" \
        --raw-root "$SCRATCH" --output-root "$KITTI" "${PUBLISH[@]}"
done

# 6. Circular top-up (manual, after reading the histogram). If ./kitti/dataset_summary.json
#    shows a class came out thin: add scenes that produce it to scene_description, add a
#    NEW split-pure run to metadata, capture that run, then mine it for only the frames
#    carrying that class. This APPENDS to the existing tree, continuing its numbering:
#
#      docker run --rm --net=host -v "$HERE":/workspace "$OFFLINE_IMAGE" \
#          python -u src/process_data/process.py --split train --raw-root "$SCRATCH" \
#          --output-root "$KITTI" --runs 6 --topup-class speed_sign_30 --topup-frames 200

# 7. The histogram is what decides whether step 6 is needed at all, so print it here
#    rather than leaving it to be found. It is written by every process invocation and
#    covers whatever splits currently exist on disk.
echo
echo "[orchestrate] ================= class histogram ================="
cat ./kitti/class_histogram.md || echo "[orchestrate] no histogram written"
echo "[orchestrate] ==================================================="

echo "[orchestrate] done."
echo "  raw    ./_scratch/scene_XXX/     (kept until this instance is terminated)"
echo "  data   ./kitti/<split>/          (image_2 label_2 calib + dataset_summary.json)"
echo "  viz    ./kitti/samples/<split>/  (annotated frames -- eyeball these)"
echo "  hist   ./kitti/class_histogram.md"
