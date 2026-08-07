"""
Capture orchestrator (Phase 1 steps 5-7).

Connects to CARLA once, captures each scene in a run via Scene + Serializer,
streams each finished scene to S3, and stamps run provenance into metadata.json.

Build order (see capture-design.md):
  Tier 1  __init__ + _capture_scene           <- subset smoke test runs on this
  Tier 2  _upload_scene + _stamp_provenance    <- needs bucket-name fix + EC2 role
  Tier 3  background upload / spot polling / resumability
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import carla
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

from config_loader import CONFIGS, CONFIG_DIR
from scene import Scene
from serializer import Serializer


# Static config bundles for the whole run.
CARLA_CFG = CONFIGS["CARLA_config.json"]
NAMING = CARLA_CFG["naming_convention"]
SENSOR_ENCODING = CARLA_CFG["sensor_encoding"]
SERVER = CARLA_CFG["server"]

# Parallel-upload width. S3 PUTs are I/O-bound, so uploading a scene's files
# concurrently is the main time win for many small objects.
_UPLOAD_WORKERS = 16


def _split_s3_bucket(uri: str) -> str:
    """'s3://bucket/some/prefix/' -> 'bucket'."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri!r}")
    return uri[len("s3://"):].split("/", 1)[0]


# Config files whose content hash goes into provenance. metadata.json is excluded --
# it is the file being written. These hashes prove which exact config produced the run.
_HASHED_CONFIGS = ("CARLA_config.json", "ego_config.json", "scene_description.json")

_IMDS = "http://169.254.169.254/latest"


def _sha256_file(path: Path) -> str:
    """SHA256 of a file's raw bytes (the on-disk config, _notes and all)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit():
    """Current commit hash, or None when not run from a git checkout (e.g. on the box)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CONFIG_DIR, capture_output=True, text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _instance_identity():
    """(region, ami_id) from IMDSv2, or (None, None) if not on EC2 / unreachable."""
    try:
        token = urllib.request.urlopen(
            urllib.request.Request(
                f"{_IMDS}/api/token", method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            ),
            timeout=2,
        ).read().decode()
        doc = json.loads(urllib.request.urlopen(
            urllib.request.Request(
                f"{_IMDS}/dynamic/instance-identity/document",
                headers={"X-aws-ec2-metadata-token": token},
            ),
            timeout=2,
        ).read())
        return doc.get("region"), doc.get("imageId")
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


class Capture:
    """One run = capture a set of scenes end-to-end and push them to S3."""

    def __init__(self, run_id, output_root, subset=None, no_upload=False):
        self.run_id = run_id
        self.output_root = Path(output_root)   # local scratch root (nvme)
        self.subset = subset                   # cap frames/scene for a subset run; None = full
        self.no_upload = no_upload             # local-only: skip S3, keep files for inspection

        # Connect the CARLA client ONCE — reused across every scene (outer lifetime).
        self.client = carla.Client(SERVER["host"], SERVER["port"])
        self.client.set_timeout(SERVER["timeout_s"])

        # Resolve this run -> its scene_ids (from metadata.json) -> scene configs, in order.
        runs = CONFIGS["metadata.json"]["runs"]
        scene_ids = next((r["scene_ids"] for r in runs if r["run_id"] == run_id), None)
        if scene_ids is None:
            raise KeyError(f"run_id {run_id} not found in metadata.json")
        scenes_by_id = {s["scene_id"]: s for s in CONFIGS["scene_description.json"]["scenes"]}
        self.scenes = [scenes_by_id[sid] for sid in scene_ids]

        # Accumulated results for provenance (timestamps are stamped by run()).
        self._scenes_completed = 0
        self._frames_captured = 0
        self._bytes_written = 0
        self._failed_scenes = []
        self._started_utc = None
        self._completed_utc = None

        # One S3 client, shared across scene uploads (boto3 clients are thread-safe).
        # Built only when uploading, so a --no-upload smoke run needs no credentials.
        # Bucket comes from metadata.json storage_root (single source of truth).
        self._s3 = None
        if not no_upload:
            self._bucket = _split_s3_bucket(CONFIGS["metadata.json"]["storage_root"])
            self._s3 = boto3.client("s3", config=Config(
                retries={"total_max_attempts": 5, "mode": "adaptive"},
                max_pool_connections=_UPLOAD_WORKERS,   # >= pool width so uploads don't queue on connections
            ))
            # Small objects (single PUT each) -> disable per-file transfer threads;
            # the file-level pool in _upload_scene provides all the concurrency.
            self._transfer_cfg = TransferConfig(use_threads=False)

    def run(self):
        """Capture every scene in the run, recording per-scene failures, then ALWAYS
        stamp provenance and push the run manifest + configs to S3 (even on abort)."""
        self._started_utc = datetime.now(timezone.utc).isoformat()
        try:
            for scene_cfg in self.scenes:
                sid = scene_cfg["scene_id"]
                try:
                    self._capture_scene(scene_cfg)
                    self._scenes_completed += 1
                except Exception as e:
                    # One bad scene shouldn't sink the other 11. Record it and move on;
                    # its S3 prefix is simply absent/partial and can be re-run later.
                    self._failed_scenes.append(sid)
                    print(f"[capture] scene {sid} FAILED: {e!r}", file=sys.stderr)
                    traceback.print_exc()
        finally:
            # Always leave provenance + a manifest, even on abort / Ctrl-C.
            self._completed_utc = datetime.now(timezone.utc).isoformat()
            self._stamp_provenance()
            self._upload_run_metadata()
        

    def _capture_scene(self, scene_cfg):
        """One scene: Scene + Serializer context, start, tick/write_frame loop, then upload."""
        with Scene(self.client, scene_cfg) as scene, Serializer(self.output_root, scene_cfg["scene_id"], NAMING, SENSOR_ENCODING) as serial:
            scene.start()
            n = scene_cfg["expected_frames"]
            if self.subset: n = min(n, self.subset)
            for _ in range(n):
                carla_frame, ts, snapshot = scene.tick()
                serial.write_frame(carla_frame, ts, snapshot)
                self._frames_captured += 1
        self._upload_scene(scene_cfg["scene_id"])

    def _upload_scene(self, scene_id):
        """
        Upload one scene's files to S3 in parallel, verify, then delete local.

        Layout: s3://<bucket>/run_<run_id:03d>/scene_<scene_id:03d>/<relpath>.
        boto3 uses the EC2 role via the default credential chain. No-op (and keeps
        the local files) when self.no_upload is set.
        """
        if self.no_upload:
            return

        scene_dir = NAMING["scene_dir"].format(scene_id=scene_id)   # "scene_001"
        local_root = self.output_root / scene_dir
        key_root = f"run_{self.run_id:03d}/{scene_dir}/"            # "run_001/scene_001/"

        # (local_path, s3_key) for every file under the scene dir -- sensor buffers
        # + records.jsonl. as_posix() keeps keys '/'-separated on Windows too.
        jobs = [
            (p, key_root + p.relative_to(local_root).as_posix())
            for p in local_root.rglob("*") if p.is_file()
        ]

        # Upload concurrently: S3 PUTs are I/O-bound, so a file-level pool is the time
        # win for many small objects. Each upload_file is a single PUT (use_threads off
        # via self._transfer_cfg), so the pool is the only concurrency. Ingress to S3
        # is free; cost is one PUT per object, unavoidable for per-frame streaming.
        def _put(job):
            local_path, key = job
            self._s3.upload_file(str(local_path), self._bucket, key, Config=self._transfer_cfg)
            return local_path.stat().st_size

        with ThreadPoolExecutor(max_workers=_UPLOAD_WORKERS) as pool:
            self._bytes_written += sum(pool.map(_put, jobs))   # map re-raises any upload error

        # Verify with ONE listing (1 request per 1000 keys) instead of N HEAD calls,
        # and only delete local once every expected key is present in S3.
        uploaded = self._list_keys(key_root)
        missing = {key for _, key in jobs} - uploaded
        if missing:
            raise RuntimeError(
                f"{scene_dir}: {len(missing)}/{len(jobs)} objects missing after upload "
                f"(e.g. {next(iter(missing))})"
            )

        shutil.rmtree(local_root)

    def _list_keys(self, prefix):
        """Set of all object keys under prefix in the run bucket (paginated)."""
        paginator = self._s3.get_paginator("list_objects_v2")
        return {
            obj["Key"]
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
        }

    def _stamp_provenance(self):
        """

        Reads the RAW metadata.json from disk (keeping its _notes -- NOT the stripped
        CONFIGS, which would drop them on write-back), fills the entry whose run_id
        matches, and writes it back. Called from run()'s finally, so it must not raise
        on a half-finished run: the server-version lookup and IMDS/git probes are all
        best-effort. The written file is what _upload_run_metadata pushes to S3.
        """
        meta_path = CONFIG_DIR / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entry = next(r for r in meta["runs"] if r["run_id"] == self.run_id)

        entry["status"] = "complete" if not self._failed_scenes else "aborted"
        entry["started_utc"] = self._started_utc
        entry["completed_utc"] = self._completed_utc

        prov = entry["provenance"]
        prov["git_commit"] = _git_commit()                 # None on the box -> hashes carry reproducibility
        for name in _HASHED_CONFIGS:                        # overwrite only the file keys, keep the _note
            prov["config_hashes"][name] = _sha256_file(CONFIG_DIR / name)
        try:
            prov["carla_version_observed"] = self.client.get_server_version()   # needs a live server
        except RuntimeError:
            prov["carla_version_observed"] = None           # server gone (e.g. crash/spot kill mid-run)
        prov["carla_python_wheel"] = self.client.get_client_version()   # local wheel, no server needed
        prov["python_version"] = platform.python_version()

        region, ami_id = _instance_identity()
        entry["compute"]["region"] = region
        entry["compute"]["ami_id"] = ami_id

        entry["results"]["scenes_completed"] = self._scenes_completed
        entry["results"]["frames_captured"] = self._frames_captured
        entry["results"]["bytes_written"] = self._bytes_written
        if self._failed_scenes:
            entry["notes"] = f"failed scenes (absent/partial in S3, re-runnable): {self._failed_scenes}"

        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta_path

    def _upload_run_metadata(self):
        """Push the stamped run manifest + the exact configs that produced it to S3.

        Layout: s3://<bucket>/run_<run_id:03d>/metadata.json (the manifest) and
        run_<run_id:03d>/config/<name>.json (every other config). Runs after
        _stamp_provenance has rewritten metadata.json. No-op when self.no_upload.
        """
        if self.no_upload:
            return
        run_root = f"run_{self.run_id:03d}/"
        # Manifest at the run root; the configs that defined the run under config/.
        for path in sorted(CONFIG_DIR.glob("*.json")):
            key = run_root + ("metadata.json" if path.name == "metadata.json"
                              else "config/" + path.name)
            self._s3.upload_file(str(path), self._bucket, key, Config=self._transfer_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA capture run")
    parser.add_argument("--run", type=int, required=True, help="run_id from metadata.json")
    parser.add_argument("--output-root", default="/opt/dlami/nvme/raw", help="local raw/ scratch root")
    parser.add_argument("--max-frames-per-scene", type=int, default=None,
                        help="subset cap: frames per scene (omit for full run)")
    parser.add_argument("--no-upload", action="store_true",
                        help="skip S3 upload and keep local files (subset smoke test)")
    args = parser.parse_args()

    Capture(args.run, args.output_root, subset=args.max_frames_per_scene,
            no_upload=args.no_upload).run()
