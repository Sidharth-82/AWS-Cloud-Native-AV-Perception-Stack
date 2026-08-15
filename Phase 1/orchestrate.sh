#!/usr/bin/env bash
#
# Tier 1 capture smoke test orchestrator.
# Starts the CARLA server (from the baked image), waits for it, builds the
# py3.7 client image, and runs the capture subset (run 1, 5 frames/scene, no upload).
#
# Run on the EC2 box from the Phase 1 dir (the one holding src/, config/, Dockerfile):
#   bash orchestrate.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SERVER_IMAGE="carlasim/carla:0.9.15"
CLIENT_IMAGE="carla-client:0.9.15"

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

# 3. Build the client image.
echo "[orchestrate] building client image ($CLIENT_IMAGE)..."
docker build -t "$CLIENT_IMAGE" -f Dockerfile .

# 4. Run the capture subset -- ONE run per map (runs 1-3 = Town04 / Town06 / Town05),
#    each its own client process. This dodges CARLA's cross-reload client-thread crash
#    (PyEval_SaveThread: NULL tstate): every load_world is a fresh process's first op, no
#    lingering sensor threads; scene.py's dedup reuses the map for the rest of the run.
#    Per-run processes also keep provenance clean -- each stamps its own metadata entry.
#    --net=host so the client reaches localhost:2000 AND the IMDS role creds. Mount
#    HERE -> /workspace so config/ resolves and output lands in ./_scratch on the host.
#    --no-upload keeps it local (Tier-1 smoke); drop it for the Tier-2 S3 verification.
echo "[orchestrate] running capture (runs 1-3 = one map each, subset=5 frames/scene) ..."
for run in 1 2 3; do
    echo "[orchestrate] --- run $run ---"
    docker run --rm --net=host -v "$HERE":/workspace "$CLIENT_IMAGE" \
        python -u src/capture.py --run "$run" \
        --output-root /workspace/_scratch --max-frames-per-scene 5 --no-upload
done

echo "[orchestrate] done. Output in ./_scratch/scene_XXX/ (records.jsonl + sensor files)."
