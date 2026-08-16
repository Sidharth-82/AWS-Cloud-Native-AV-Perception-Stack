
#################################
"""
This file is meant to house code dedicated towards parsing

Functions within:

def iter_run_frames(run_id, configs, buffers=...) -> Iterator[(record, dict)]
    - Lazily streams ONE run's frames from S3 one at a time: yields a per-frame
      record and a dict of DECODED sensor buffers. Bounded prefetch, nothing
      held on disk. This is the entry point for offline step 8.

def iter_cycle_frames(run_ids, configs, buffers=...) -> Iterator[(record, dict)]
    - Same stream, across SEVERAL runs back to back. A capture cycle is split
      into multiple runs (one client process per map), and splits cut across
      them, so the offline passes almost always want this one.

def label_frame(record, buffers, configs) -> (lines, track_ids, stats)
    - Phase 1 step 8 for one frame: project every recorded box into the image,
      cull what the camera cannot actually see, and emit KITTI label lines for
      what survives, each with its ground-truth track id. Returns drop counts
      by reason alongside the labels.

def write_kitti(output_root, run_ids, ...) -> stats
    - Phase 1 step 9: stream a run set and write one flat KITTI tree
      (image_2/label_2/calib[/velodyne] + frame_index.json). One invocation
      writes one tree; splits are made by invoking it per run set.

"""


### Imports below

from collections.abc import Iterator
import io
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, product
import shutil
from pathlib import Path
import random


import boto3
import numpy as np
from botocore.config import Config
from PIL import Image

from config_loader import CONFIGS
from utils import RECORDS_FILENAME


### S3 streaming below

# Maps each sensor-buffer name (the record's sensor_files keys) to its decoder.
# Depth and instance-seg are ENCODED buffers, not literal images -- see
# CARLA_config.sensor_encoding. RGB and lidar decode literally.
_DEPTH_FAR_M = 1000        # CARLA depth far plane (sensor_encoding.cam_front_depth)
_DEPTH_24BIT_MAX = 256**3 - 1   # full-scale value of the 24-bit packed depth

# Channel layout of the array _decode_instance_seg returns (HxWx2 uint16).
SEG_TAG = 0        # semantic tag: 14=Car, 15=Truck, 16=Bus, 18=Motorcycle, 8=TrafficSign, ...
SEG_INSTANCE = 1   # opaque per-object separator -- NOT carla actor.id, see below


def _decode_rgb(raw: bytes) -> np.ndarray:
    """Front RGB: literal PNG -> HxWx3 uint8."""
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _decode_depth(raw: bytes) -> np.ndarray:
    """
    CARLA depth: 24-bit depth packed in RGB -> HxW float32 metres.

    depth_m = 1000 * (R + G*256 + B*256^2) / (256^3 - 1)   [CARLA_config]

    VERIFIED on the pulled subset: highway front cam gives min ~2.8 m, median
    ~31 m, with ~27% of pixels sitting at the 1000 m far plane (sky//beyond
    range). Pixels AT the far plane mean "nothing here", not "an object 1 km
    away" -- the occlusion check must treat them as empty, not as a hit.

    Channels are widened to uint32 before packing: the PNG decodes to uint8, so
    G*256 would silently wrap to zero.
    """
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.uint32)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    packed = R + G * 256 + B * 256**2
    return (_DEPTH_FAR_M * (packed / _DEPTH_24BIT_MAX)).astype(np.float32)


def _decode_instance_seg(raw: bytes) -> np.ndarray:
    """
    CARLA instance-seg -> HxWx2 uint16: [..., SEG_TAG] and [..., SEG_INSTANCE].

    R is the semantic tag (14=Car, 15=Truck, 16=Bus, 18=Motorcycle,
    8=TrafficSign, ...); G/B pack a 16-bit instance id as (G*256) + B.

    The instance id is NOT carla actor.id -- verified empirically on the subset:
    vehicle-tag pixels decode to ids like 19409/62932 under BOTH byte orders,
    while the recorded actor ids for those frames were 277-306. It is an opaque
    engine-side id with no route back to the Python actor, which is why the
    visibility filter uses DEPTH as its occlusion oracle. Treat this channel
    only as a separator between two same-class vehicles whose projected boxes
    overlap, and the tag channel as the class confirmation.

    Both channels fit uint16 (tags < 256, ids <= 65535), so they stack losslessly.
    """
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.uint16)
    tag = img[:, :, 0]
    instance = img[:, :, 1] * 256 + img[:, :, 2]

    return np.stack((tag, instance), axis=-1)


def _decode_lidar(raw: bytes) -> np.ndarray:
    """LiDAR: .npy float32 (N, 4) = x, y, z, intensity."""
    return np.load(io.BytesIO(raw))


_DECODERS = {
    "cam_front": _decode_rgb,
    "cam_front_depth": _decode_depth,
    "cam_front_instance_seg": _decode_instance_seg,
    "lidar_top": _decode_lidar,
}


# S3 layout -- WRITTEN by capture.Capture._upload_scene, READ here. The two must
# agree; if one moves, move the other.
#
#   run_001/                            <- one run = one client process = one map
#     metadata.json                     <- that run's stamped manifest
#     config/*.json                     <- the exact configs that produced it
#     scene_001/
#       records.jsonl                   <- one JSON record per captured frame
#       cam_front/001_231495.png
#       cam_front_depth/001_231495.png
#       cam_front_instance_seg/001_231495.png
#       lidar_top/001_231495.npy
#
# A record's sensor_files values are ALREADY 'scene_XXX/<sensor>/<frame>.<ext>',
# so a full object key is just the run prefix + that value -- no per-scene prefix
# lookup, and nothing to keep in sync but the run number.


def _run_prefix(run_id) -> str:
    """run 1 -> 'run_001/'. Mirrors capture._upload_scene's key_root."""
    return f"run_{run_id:03d}/"


class LocalSource:
    """
    Read raw frames from a capture scratch directory on this machine.

    The DEFAULT source. Capture and processing run on the same box, so raw
    frames never need to leave it -- only the finished KITTI tree is published.

    Note the layout difference this class exists to absorb: capture writes
    locally as '<root>/scene_001/...' with NO run prefix (it only prepends
    'run_001/' when uploading), so the local key prefix is empty. Scene ids are
    globally unique, so scenes from several runs share one scratch root safely.
    """

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"raw scratch root does not exist: {self.root}")

    def key_prefix(self, run_id) -> str:
        return ""

    def get(self, key: str) -> bytes:
        path = self.root / key
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found in the local scratch root")
        return path.read_bytes()


class S3Source:
    """
    Read raw frames from the bucket, for runs captured with --upload-raw.

    Not the normal path any more, but kept: it is what lets an archived run be
    reprocessed off-box, and it is the only source that works if the capture
    instance is already gone.
    """

    def __init__(self, configs: dict, max_workers: int = 16):
        self.bucket = _bucket(configs)
        self.s3 = boto3.client("s3", config=Config(
            retries={"total_max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=max_workers,
        ))

    def key_prefix(self, run_id) -> str:
        return _run_prefix(run_id)

    def get(self, key: str) -> bytes:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"s3://{self.bucket}/{key} not found") from None


def make_source(raw_root=None, configs: dict = CONFIGS, max_workers: int = 16):
    """LocalSource when raw_root is given (the default flow), else S3Source."""
    return LocalSource(raw_root) if raw_root is not None else S3Source(configs, max_workers)


def _bucket(configs: dict) -> str:
    """
    Bucket name from metadata.storage_root ('s3://bucket/' -> 'bucket').

    Any path component after the bucket is ignored, because the writer ignores
    it too: capture builds keys as 'run_XXX/...' from the bucket root.
    """
    root = configs["metadata.json"]["storage_root"]
    if not root.startswith("s3://"):
        raise ValueError(f"storage_root is not an s3 uri: {root!r}")
    return root[len("s3://"):].split("/", 1)[0]


def _run_scene_ids(configs: dict, run_id) -> list:
    """Resolve run_id -> its scene_ids via metadata.json."""
    for run in configs["metadata.json"]["runs"]:
        if run["run_id"] == run_id:
            return run["scene_ids"]
    raise KeyError(f"run_id {run_id!r} not found in metadata.json")


def _complete_run_ids(configs: dict) -> list:
    """
    Every run_id whose capture actually finished, in metadata order.

    A run is only streamable once its scenes are in S3, so 'planned' and
    'in_progress' runs are skipped. 'aborted' is skipped too: it has holes
    where scenes failed, and a missing records.jsonl would surface as a
    mid-stream 404 rather than an honest error. Stream one explicitly (by id)
    if you want its surviving scenes.
    """
    runs = configs["metadata.json"]["runs"]
    complete = [r["run_id"] for r in runs if r.get("status") == "complete"]
    if not complete:
        raise RuntimeError(
            "no runs are marked complete in metadata.json (statuses: "
            + ", ".join(f"run {r['run_id']}={r.get('status')!r}" for r in runs)
            + "). Pass run_ids explicitly to stream one anyway."
        )
    return complete


def _iter_fetch_jobs(source, run_id, scene_ids: list, naming: dict, frame_ids=None,
                     exclude_frame_ids=None):
    """
    Walk a run's scenes and yield one fetch job per frame.

    For each scene: ONE read pulls that scene's records.jsonl, then a job is
    emitted per record. A job is (key_prefix, record) -- everything the worker
    needs to build the per-buffer keys, since key_prefix + record's
    sensor_files[name] IS the key.

    Scenes are walked in metadata scene_ids order, and each scene's records
    file is read lazily -- only when the consumer reaches that scene.

    frame_ids filters at the JOB level rather than after fetching, so a
    top-up pass that wants 200 specific frames does not pay to read the
    buffers of the thousands it is going to discard. exclude_frame_ids is the
    same mechanism in reverse, used to skip frames already in the tree.
    """
    key_prefix = source.key_prefix(run_id)
    wanted = set(frame_ids) if frame_ids is not None else None
    skip = set(exclude_frame_ids) if exclude_frame_ids else set()
    for scene_id in scene_ids:
        scene_dir = naming["scene_dir"].format(scene_id=scene_id)
        records_key = f"{key_prefix}{scene_dir}/{RECORDS_FILENAME}"
        try:
            body = source.get(records_key)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e} -- scene {scene_id} of run {run_id} was never captured, or the "
                "run aborted before writing it."
            ) from None
        for line in body.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["frame_id"] in skip:
                continue
            if wanted is None or record["frame_id"] in wanted:
                yield key_prefix, record


def _fetch_frame(source, key_prefix: str, record: dict, buffers: tuple):
    """
    Read + decode one frame's requested buffers. Runs on a pool thread.

    One read per buffer -- on the S3 source that per-frame request count is the
    real cost of the stream, which is why `buffers` defaults to just the two the
    filter needs.
    """
    decoded = {}
    for name in buffers:
        try:
            rel = record["sensor_files"][name]
        except KeyError:
            raise KeyError(
                f"frame {record['frame_id']} has no {name!r} buffer; "
                f"captured: {sorted(record['sensor_files'])}"
            ) from None
        decoded[name] = _DECODERS[name](source.get(key_prefix + rel))
    return record, decoded


def iter_run_frames(
    run_id,
    configs: dict = CONFIGS,
    buffers: tuple = ("cam_front_instance_seg", "cam_front_depth"),
    prefetch: int = 64,
    max_workers: int = 16,
    scene_ids=None,
    raw_root=None,
    frame_ids=None,
    exclude_frame_ids=None,
) -> Iterator[tuple[dict, dict[str, np.ndarray]]]:
    """
    Lazily stream ONE run's frames, one at a time.

    Yields (record, buffers) per frame, where record is the parsed per-frame
    dict (schema = dataset_example.json) and buffers maps each requested buffer
    name to its DECODED numpy array. Encoded depth/instance-seg are decoded
    before yielding -- raw bytes never surface.

    run_id is FIXED for the life of the generator: one generator walks one run's
    frames (scene by scene, in scene_ids order) then stops. It cannot be handed
    a new run_id mid-iteration. A capture cycle spans several runs, so for the
    usual whole-dataset pass call iter_cycle_frames instead.

    Sensor bytes are fetched by a bounded ThreadPoolExecutor so network I/O
    overlaps the caller's per-frame CPU work; at most `prefetch` frames are in
    flight, so memory stays flat and nothing is written to disk. Frames are
    yielded in submission order (deterministic -- handy for the sanity-viz),
    which the visibility filter does not depend on.

    Args:
        run_id: which run (metadata.runs[].run_id) to stream.
        configs: the config bundle (defaults to the module CONFIGS).
        buffers: sensor buffers to fetch + decode per frame; names are the
            record's sensor_files keys. Default is the pair the visibility
            filter needs -- add "cam_front" / "lidar_top" only when required
            (e.g. RGB for the ~10 sanity-viz frames), since each extra buffer
            is another GET per frame.
        prefetch: max frames in flight (bounds memory).
        max_workers: number of reader threads.
        scene_ids: restrict to these scenes (intersected with the run's own, in
            the run's order). None streams the whole run.
        raw_root: local capture scratch directory to read from -- the normal
            flow, since capture and processing share a box. None falls back to
            reading the bucket, for runs captured with --upload-raw.
        frame_ids: only these frame_ids, filtered before any buffer is read.
            Used by the top-up pass to fetch just its sampled frames.

    Yields:
        (record: dict, buffers: dict[str, np.ndarray])
    """
    unknown = tuple(name for name in buffers if name not in _DECODERS)
    if unknown:
        raise KeyError(f"no decoder for buffer(s) {unknown}; known: {tuple(_DECODERS)}")
    if prefetch < 1 or max_workers < 1:
        raise ValueError("prefetch and max_workers must both be >= 1")

    run_scenes = _run_scene_ids(configs, run_id)
    if scene_ids is not None:
        wanted = set(scene_ids)
        run_scenes = [s for s in run_scenes if s in wanted]
    naming = configs["CARLA_config.json"]["naming_convention"]

    # One source for the whole generator. The S3 flavour holds a boto3 client,
    # which is thread-safe and pool-sized to the fetch pool so workers never
    # queue on sockets; the local flavour is a plain directory read.
    source = make_source(raw_root, configs, max_workers)

    jobs = _iter_fetch_jobs(source, run_id, run_scenes, naming, frame_ids,
                            exclude_frame_ids)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        def submit(job):
            return pool.submit(_fetch_frame, source, job[0], job[1], buffers)

        # Sliding window: prime it with `prefetch` frames, then hold that depth by
        # submitting one more each time the caller consumes one. Popping the LEFT
        # future is what makes the output submission-ordered -- and it doubles as
        # the backpressure, since nothing new is fetched until a frame is taken.
        window = deque(submit(job) for job in islice(jobs, prefetch))
        for job in jobs:
            yield window.popleft().result()
            window.append(submit(job))
        while window:
            yield window.popleft().result()


def iter_cycle_frames(
    run_ids=None,
    configs: dict = CONFIGS,
    buffers: tuple = ("cam_front_instance_seg", "cam_front_depth"),
    prefetch: int = 64,
    max_workers: int = 16,
    scene_ids=None,
    raw_root=None,
    frame_ids=None,
    exclude_frame_ids=None,
) -> Iterator[tuple[dict, dict[str, np.ndarray]]]:
    """
    Stream SEVERAL runs back to back as one flat frame stream.

    A capture cycle is deliberately split into multiple runs -- one client
    process per map, since CARLA's client threads crash on a world reload after
    a prior scene's sensors ran -- and train/val/test cut ACROSS those runs. So
    the dataset-wide passes (visibility filter, KITTI writer, histogram) want
    every run's frames, not one run's. Runs are streamed in order, one at a
    time, so only one run's prefetch window is ever in flight.

    Each frame's record carries its own scene_id, and scene_description assigns
    the split per scene -- so the consumer routes frames to splits from the
    record, and never needs to know which run a frame came from.

    Args:
        run_ids: which runs to stream, in order. Defaults to every run marked
            'complete' in metadata.json (see _complete_run_ids); pass an
            explicit list to stream a subset, or to force an aborted run.
        configs, buffers, prefetch, max_workers, scene_ids: as iter_run_frames.
            scene_ids is how one split is streamed out of a run set, since a
            single run holds scenes from more than one split.

    Yields:
        (record: dict, buffers: dict[str, np.ndarray])
    """
    if run_ids is None:
        run_ids = _complete_run_ids(configs)
    for run_id in run_ids:
        yield from iter_run_frames(run_id, configs, buffers, prefetch, max_workers,
                                   scene_ids, raw_root, frame_ids, exclude_frame_ids)


### Geometry below

# All of the following was resolved EMPIRICALLY against the pulled subset, not
# assumed. Each one is a silent-wrong-label bug if guessed wrong:
#
#   * cam_front_depth is PLANAR (distance along the camera's forward axis), not
#     radial. Checked by projecting LiDAR points into the camera and comparing:
#     buffer/planar = 1.0000, buffer/radial = 0.88.
#   * The LiDAR .npy needs NO y-flip -- its points are already in the same
#     left-handed sensor frame as the camera (91-96% per-point depth agreement
#     vs 37-52% when flipped).
#   * fx = fy = 640, cx = 640, cy = 360 for the 1280x720 90deg rig, and the
#     CARLA->pinhole axis map is (y, -z, x). The 1.0000 ratio above only holds
#     if both are right.
#
# Frames, in order: CARLA world (left-handed, x fwd / y right / z up)
#   -> ego -> CARLA camera (same axes) -> pinhole aka KITTI camera (x right /
#   y down / z fwd) -> pixels.

_EPS_Z_M = 0.1          # a corner must be at least this far in front to project

# CARLA semantic tags, read from the instance-seg R channel. Rider (13) is in the
# VEHICLE set on purpose: CARLA tags a motorcycle's rider separately from the bike
# but gives them the SAME instance id, so leaving 13 out silently discarded a third
# of every motorcycle's pixels -- enough to drop the whole class from the dataset.
VEHICLE_TAGS = (13, 14, 15, 16, 18)   # Rider, Car, Truck, Bus, Motorcycle
SIGN_TAG = 8                          # TrafficSign

# Visibility knobs. Tunable; candidates to move into CARLA_config.json next to
# class_map if they ever need to vary per dataset.
MIN_UNOCCLUDED_FRAC = 0.4   # see _visible_pixels: object px / (object px + blocking px)
MIN_VISIBLE_PIXELS = 60     # absolute floor for vehicles, independent of the fraction
MIN_SIGN_PIXELS = 15        # signs are small; a 0.68 m plate at 60 m is tiny
MIN_BOX_HEIGHT_PX = 20      # vehicles. KITTI's own 'easy' tier starts at 40 px

# Signs need their OWN floor, not the vehicle one. A 0.68 m plate only reaches
# 20 px at 22 m, so the vehicle floor was discarding perfectly readable signs --
# and since the class IS the posted number, the floor has to be set where the
# digits stay legible. Checked by cropping real signs at each size: crisp at
# 30 px (15 m), still legible at 21 px (20 m), marginal at 15 px (30 m), gone
# below. 18 px (~24 m) sits just inside legible.
MIN_SIGN_HEIGHT_PX = 18
DEPTH_TOL_M = 0.75          # slack around the box's own depth span
MAX_RANGE_M = 150.0         # beyond this a highway vehicle is a few pixels

# Sign boxes are paper-thin in local y (verified: 77/77 signs in the subset have
# y as their smallest half-extent), so the readable face's normal is local +/-y.
# This picks which of the two. See _sign_faces_camera.
_SIGN_FACE_SIGN = 1.0


def _xyz(d: dict) -> np.ndarray:
    """{'x':..,'y':..,'z':..} -> array([x, y, z])."""
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)


def _intrinsics(configs: dict):
    """(f, cx, cy, width, height) DERIVED from width/height/fov, per ego_config."""
    cam = configs["ego_config.json"]["sensor_rig"]["sensors"]["cam_front"]
    w, h = int(cam["width"]), int(cam["height"])
    f = w / (2.0 * np.tan(np.radians(cam["fov_deg"]) / 2.0))
    return f, w / 2.0, h / 2.0, w, h


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    CARLA rotation (DEGREES) -> 3x3 local->world matrix.

    This is CARLA's own composition order (Transform::GetMatrix), not a generic
    RPY: getting the order wrong tilts every box in a way that looks almost
    right at small angles and is badly wrong in turns.
    """
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,      -cp * sr,                cp * cr],
    ])


def _camera_pose(ego_pose: dict, configs: dict):
    """
    (R, t) taking world -> CARLA camera frame, as p_cam = R @ (p_world - t).

    t is the camera's world position; R is the inverse of its world rotation.
    Composed from ego_pose and T_sensor_to_ego, per coordinate_conventions.
    """
    R_ego = _rotation_matrix(**ego_pose["rotation"])
    t_ego = _xyz(ego_pose["position"])
    ext = configs["ego_config.json"]["sensor_rig"]["sensors"]["cam_front"]["T_sensor_to_ego"]
    R_cam_ego = _rotation_matrix(**ext["rotation"])
    return (R_cam_ego.T @ R_ego.T), (t_ego + R_ego @ _xyz(ext["position"]))


def _to_pinhole(p_cam: np.ndarray) -> np.ndarray:
    """CARLA camera axes (x fwd, y right, z up) -> KITTI camera (x right, y down, z fwd)."""
    return np.stack([p_cam[..., 1], -p_cam[..., 2], p_cam[..., 0]], axis=-1)


def _box_corners_world(bbox: dict) -> np.ndarray:
    """The 8 world-frame corners of a recorded bounding_box_world. (8, 3)."""
    center, extent = _xyz(bbox["center"]), _xyz(bbox["half_extent"])
    R = _rotation_matrix(**bbox["rotation"])
    signs = np.array(list(product((-1.0, 1.0), repeat=3)))
    return center + (signs * extent) @ R.T


def _hull_area(pts: np.ndarray) -> float:
    """
    Convex-hull area of a small 2D point set (monotone chain + shoelace).

    Used as the object's SILHOUETTE proxy. The axis-aligned box would overstate
    it badly for a rotated vehicle -- a box at 45 degrees is ~half background --
    which would make every turning car look occluded.
    """
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def _chain(points):
        out = []
        for q in points:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) > 0:
                    break
                out.pop()
            out.append(tuple(q))
        return out

    hull = np.array(_chain(p)[:-1] + _chain(p[::-1])[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _project_object(obj: dict, R: np.ndarray, t: np.ndarray, intr, min_height: float = MIN_BOX_HEIGHT_PX) -> dict:
    """
    Project one object's 3D box into the image.

    Returns None when the box is entirely behind the camera, out of frame, or
    past MAX_RANGE_M. Otherwise: the clipped 2D box, KITTI truncation, the
    silhouette area used as the visibility denominator, and the box's own depth
    span (the occlusion test compares rendered depth against this).

    Corners behind the camera are dropped rather than projected -- a negative z
    would fling them to the wrong side of the image. That slightly understates
    the box of an object straddling the image plane, which is exactly the case
    KITTI marks as heavily truncated anyway.
    """
    f, cx, cy, W, H = intr
    corners_pin = _to_pinhole((_box_corners_world(obj["bounding_box_world"]) - t) @ R.T)

    front = corners_pin[:, 2] > _EPS_Z_M
    if not front.any():
        return None
    vis = corners_pin[front]
    depth_near, depth_far = float(vis[:, 2].min()), float(vis[:, 2].max())
    if depth_near > MAX_RANGE_M:
        return None

    uv = np.stack([f * vis[:, 0] / vis[:, 2] + cx, f * vis[:, 1] / vis[:, 2] + cy], axis=-1)
    full_area = _hull_area(uv)

    # Clamping corners into the frame approximates clipping the silhouette: both
    # the truncation ratio and the visibility denominator come from the result.
    clamped = np.stack([np.clip(uv[:, 0], 0, W), np.clip(uv[:, 1], 0, H)], axis=-1)
    box = (clamped[:, 0].min(), clamped[:, 1].min(), clamped[:, 0].max(), clamped[:, 1].max())
    if (box[2] - box[0]) < 1.0 or (box[3] - box[1]) < min_height:
        return None

    silhouette = _hull_area(clamped)
    if silhouette <= 0.0:
        return None

    return {
        "box2d": box,
        "truncation": float(np.clip(1.0 - silhouette / full_area, 0.0, 1.0)) if full_area > 0 else 0.0,
        "silhouette_px": silhouette,
        "depth_near": depth_near,
        "depth_far": depth_far,
        "corners_pin": corners_pin,
    }


def _sign_faces_camera(obj: dict, cam_pos: np.ndarray) -> bool:
    """
    True when the sign's READABLE face points at the camera.

    A speed sign on the opposite carriageway still projects into frame and still
    passes an occlusion test, but its pixels are a blank grey back -- labeling it
    'speed_sign_90' teaches the detector that the back of a sign posts 90.
    """
    bbox = obj["bounding_box_world"]
    normal = _SIGN_FACE_SIGN * _rotation_matrix(**bbox["rotation"])[:, 1]
    return float(normal @ (cam_pos - _xyz(bbox["center"]))) > 0.0


def _visible_pixels(view: dict, depth: np.ndarray, seg: np.ndarray, tags, is_sign: bool):
    """
    Measure how much of this object the camera can actually see.

    A pixel is the OBJECT's when its semantic tag matches the object's kind and
    its rendered depth falls inside the box's own depth span. Depth is the
    occlusion oracle because the instance-seg id is opaque engine-side and cannot
    be matched back to a carla actor id (verified on the subset). For vehicles
    the opaque id still earns its keep as a SEPARATOR: only pixels sharing the
    dominant id are counted, so two same-class cars at similar range whose boxes
    overlap don't credit each other.

    The fraction is object_px / (object_px + blocking_px), where blocking pixels
    are those rendered strictly IN FRONT of the box. It is deliberately not
    'object px / projected box area': a box is mostly empty space for a thin
    object, so that ratio caps a perfectly visible motorcycle at ~0.25 and caps
    a rotated car well under 1, which biases the filter against exactly the
    classes that are hardest to collect. Measuring occluders instead makes the
    number mean what it says -- an unobstructed object scores ~1.0 whatever its
    shape -- and leaves background and road pixels correctly neutral, since they
    are neither the object nor in front of it.

    Returns (object_pixels, unoccluded_fraction).
    """
    x0, y0, x1, y1 = view["box2d"]
    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    x1, y1 = int(np.ceil(x1)), int(np.ceil(y1))
    d = depth[y0:y1, x0:x1]
    s = seg[y0:y1, x0:x1]
    if d.size == 0:
        return 0, 0.0

    # Far-plane pixels are 'nothing rendered here', not 'an object 1 km away'.
    rendered = d < _DEPTH_FAR_M - 1.0
    obj = (
        rendered
        & np.isin(s[..., SEG_TAG], tags)
        & (d >= view["depth_near"] - DEPTH_TOL_M)
        & (d <= view["depth_far"] + DEPTH_TOL_M)
    )
    if not is_sign and obj.any():
        obj &= s[..., SEG_INSTANCE] == np.bincount(s[..., SEG_INSTANCE][obj]).argmax()

    n = int(obj.sum())
    if n == 0:
        return 0, 0.0
    blocking = int((rendered & (d < view["depth_near"] - DEPTH_TOL_M)).sum())
    return n, n / (n + blocking)


def _occlusion_level(frac: float) -> int:
    """KITTI occlusion code from the unoccluded fraction: 0 fully, 1 partly, 2 largely."""
    if frac >= 0.9:
        return 0
    if frac >= 0.6:
        return 1
    return 2


### Labeling below

def _class_of(obj: dict, class_map: dict):
    """
    Dataset class for one recorded object, via class_map's ACTIVE preset.

    Returns None when the object has no mapping -- an unmapped base_type or a
    posted speed with no class. Callers count those rather than guessing.
    """
    preset = class_map["presets"][class_map["active_preset"]]
    if obj["source"] == "carla_actor":
        return preset["vehicle_by_base_type"].get(obj["base_type"])
    speed = obj.get("listed_speed_kph")
    if speed is None:
        return None
    return preset["sign_by_speed_kph"].get(str(int(round(speed))))


def _label_line(cls: str, obj: dict, view: dict, occlusion: int, R: np.ndarray, t: np.ndarray) -> str:
    """
    One KITTI label_2 line: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry.

    Conversions worth stating, since each is a classic silent bug:
      * dimensions are (h, w, l) = 2 * half_extent (z, y, x) -- CARLA stores
        HALF extents with x along the object's length.
      * location is the box's BOTTOM-centre in camera coords, not its centroid
        (KITTI's devkit builds corners with y in [0, -h]).
      * rotation_y is read off the object's forward axis after the same
        world->camera rotation everything else uses, rather than from a yaw
        subtraction, so a non-zero ego pitch/roll can't quietly corrupt it.
    """
    bbox = obj["bounding_box_world"]
    extent = _xyz(bbox["half_extent"])
    h, w, length = 2 * extent[2], 2 * extent[1], 2 * extent[0]

    centre = _to_pinhole((_xyz(bbox["center"]) - t) @ R.T)
    x, y, z = centre[0], centre[1] + h / 2.0, centre[2]

    forward = _to_pinhole(R @ _rotation_matrix(**bbox["rotation"])[:, 0])
    ry = float(np.arctan2(-forward[2], forward[0]))
    alpha = float(np.arctan2(np.sin(ry - np.arctan2(x, z)), np.cos(ry - np.arctan2(x, z))))

    x1, y1, x2, y2 = view["box2d"]
    return (
        f"{cls} {view['truncation']:.2f} {occlusion} {alpha:.2f} "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.2f} {w:.2f} {length:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
    )


def label_frame(record: dict, buffers: dict, configs: dict = CONFIGS):
    """
    Turn one streamed frame into KITTI label lines, keeping only visible objects.

    This is Phase 1 step 8 end to end for a single frame: project every recorded
    box, cull what the camera cannot actually see, and label what survives.
    Capture deliberately records every actor in the map -- ~120 vehicles and ~59
    signs per frame on Town04 -- so the great majority is expected to be dropped
    here, and the drop reasons are returned rather than swallowed.

    Args:
        record: one frame record from the stream.
        buffers: must include 'cam_front_depth' and 'cam_front_instance_seg'.
        configs: the config bundle.

    Returns:
        (lines, track_ids, stats). track_ids is POSITIONALLY ALIGNED with lines:
        track_ids[i] identifies the object labelled on lines[i]. KITTI's detection
        format has no identity column -- and its 16th field is conventionally the
        confidence score, so an id put there would be read as a score -- hence the
        parallel list, carried in frame_index.json rather than in the label file.
        stats counts kept objects per class and drops per reason, so a
        systematically-vanishing class is visible rather than silently absent
        from the histogram.
    """
    depth = buffers["cam_front_depth"]
    seg = buffers["cam_front_instance_seg"]
    intr = _intrinsics(configs)
    class_map = configs["CARLA_config.json"]["class_map"]
    R, t = _camera_pose(record["ego_pose"], configs)

    lines, track_ids, kept, dropped = [], [], {}, {}

    def _drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    # A preset with no sign classes excludes signs DELIBERATELY. Reported under its
    # own reason rather than as 'unmapped_class', which would put thousands of
    # intentional exclusions under a label that means "this object fell through a
    # gap in the class map" -- the one stat that is supposed to expose real bugs.
    preset = class_map["presets"][class_map["active_preset"]]
    signs_excluded = not preset["sign_by_speed_kph"]

    for obj in record["actors"]:
        is_sign = obj["source"] != "carla_actor"
        if is_sign and signs_excluded:
            _drop("signs_excluded")
            continue
        cls = _class_of(obj, class_map)
        if cls is None:
            _drop("unmapped_class")
            continue

        # Project BEFORE the facing test, so 'sign_facing_away' counts only signs
        # that were actually in view -- otherwise it absorbs every sign in the map
        # and the stat says nothing about what the rule costs.
        view = _project_object(obj, R, t, intr,
                               MIN_SIGN_HEIGHT_PX if is_sign else MIN_BOX_HEIGHT_PX)
        if view is None:
            _drop("out_of_frustum")
            continue
        if is_sign and not _sign_faces_camera(obj, t):
            _drop("sign_facing_away")
            continue

        tags = (SIGN_TAG,) if is_sign else VEHICLE_TAGS
        n_px, frac = _visible_pixels(view, depth, seg, tags, is_sign)
        if n_px < (MIN_SIGN_PIXELS if is_sign else MIN_VISIBLE_PIXELS):
            _drop("too_few_pixels")
            continue
        if frac < MIN_UNOCCLUDED_FRAC:
            _drop("occluded")
            continue

        # Appended together, so the two lists cannot drift out of alignment.
        # global_actor_id is scene-scoped ("002_2673"), which is what makes it a
        # usable track id: CARLA actor ids are stable for an actor's lifetime
        # within an episode, and the scene prefix keeps ids from colliding across
        # scenes that reuse the same numbers.
        lines.append(_label_line(cls, obj, view, _occlusion_level(frac), R, t))
        track_ids.append(obj["global_actor_id"])
        kept[cls] = kept.get(cls, 0) + 1

    return lines, track_ids, {"kept": kept, "dropped": dropped}


### KITTI packaging below

def _calib_text(configs: dict) -> str:
    """
    KITTI calib block for this rig.

    P0..P3 are identical: this is a single-camera rig, so there is no stereo
    baseline to encode, and KITTI readers take P2. R0_rect is identity because
    CARLA cameras are ideal pinholes with nothing to rectify.

    Tr_velo_to_cam is composed, not hand-written: velodyne (KITTI: x fwd, y left,
    z up) -> CARLA LiDAR frame (y right) -> ego -> camera -> KITTI camera axes.
    """
    f, cx, cy, _, _ = _intrinsics(configs)
    sensors = configs["ego_config.json"]["sensor_rig"]["sensors"]
    cam, lidar = sensors["cam_front"]["T_sensor_to_ego"], sensors["lidar_top"]["T_sensor_to_ego"]

    P = np.array([[f, 0.0, cx, 0.0], [0.0, f, cy, 0.0], [0.0, 0.0, 1.0, 0.0]])
    M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])   # carla cam -> kitti cam
    F = np.diag([1.0, -1.0, 1.0])                                        # kitti velo -> carla lidar
    R_cam_ego = _rotation_matrix(**cam["rotation"])
    R_lidar_ego = _rotation_matrix(**lidar["rotation"])

    rot = M @ R_cam_ego.T @ R_lidar_ego @ F
    trans = M @ R_cam_ego.T @ (_xyz(lidar["position"]) - _xyz(cam["position"]))
    tr = np.hstack([rot, trans.reshape(3, 1)])

    def _row(name, mat):
        return name + ": " + " ".join(f"{v:.12e}" for v in np.asarray(mat).ravel())

    return "\n".join([
        *(_row(f"P{i}", P) for i in range(4)),
        _row("R0_rect", np.eye(3)),
        _row("Tr_velo_to_cam", tr),
        _row("Tr_imu_to_velo", np.hstack([np.eye(3), np.zeros((3, 1))])),
    ]) + "\n"


def _select_frames(select_class, select_n, seed, run_ids, configs, scene_ids,
                   raw_root, prefetch, max_workers, already: set) -> set:
    """
    SCAN pass: find the frames whose labels contain select_class, then sample.

    Two passes rather than one, because sampling needs to see every candidate
    before it can choose, and buffering whole frames to decide later would mean
    holding megabytes of decoded image per candidate. The scan reads only depth
    and instance-seg -- never the RGB -- so it costs a fraction of a write pass,
    and the write pass then fetches buffers for the sampled frames alone.

    Sampling matters: a re-captured 300 s scene yields thousands of frames
    containing the class, nearly all near-duplicates of their neighbours at
    2 Hz. Taking a random spread is what makes a top-up add information rather
    than weight.
    """
    matches = []
    scan = iter_cycle_frames(run_ids, configs,
                             ("cam_front_instance_seg", "cam_front_depth"),
                             prefetch, max_workers, scene_ids, raw_root,
                             None, already)
    for record, bufs in scan:
        _, _, stats = label_frame(record, bufs, configs)
        if stats["kept"].get(select_class):
            matches.append(record["frame_id"])

    if select_n is not None and len(matches) > select_n:
        rng = random.Random(seed)          # seeded: a top-up round is reproducible
        matches = rng.sample(matches, select_n)
    print(f"[write_kitti] {select_class}: {len(matches)} frames selected")
    return set(matches)


def write_kitti(
    output_root,
    run_ids=None,
    configs: dict = CONFIGS,
    include_lidar: bool = True,
    limit: int = None,
    prefetch: int = 64,
    max_workers: int = 16,
    scene_ids=None,
    raw_root=None,
    append: bool = False,
    select_class: str = None,
    select_n: int = None,
    select_seed: int = 0,
) -> dict:
    """
    Stream a set of runs and write ONE flat KITTI tree (Phase 1 step 9).

        <output_root>/image_2/000000.png     label_2/000000.txt
                      calib/000000.txt      [velodyne/000000.bin]
                      frame_index.json

    Splits are NOT handled here. One invocation = one tree; point it at the run
    set for a split and run it again for the next. That keeps the writer unable
    to leak frames across splits, since it never sees two splits at once.

    Frames are renumbered sequentially per invocation because KITTI tooling
    assumes %06d stems. frame_index.json maps each stem back to its
    (run_id, scene_id, carla_frame, frame_id), so nothing loses its provenance.

    Args:
        output_root: directory to write the tree into (created if absent).
        run_ids: runs to include; defaults to every 'complete' run.
        include_lidar: also write velodyne/*.bin. ON by default: the sweep is
            captured regardless, so omitting it from the published dataset just
            throws away data that cost GPU time to make. Costs one extra read
            per frame and roughly 0.5 MB per frame on disk.
        limit: stop after this many frames (subset runs / smoke tests).
        scene_ids: restrict to these scenes within the selected runs.
        raw_root: local capture scratch to read (the normal flow); None reads S3.
        append: extend an existing tree instead of starting at 000000. Numbering
            continues from the tree's highest stem and frame_index.json is
            merged, so a top-up round grows the dataset in place.
        select_class: keep only frames whose labels contain this class. This is
            the top-up filter -- after the histogram shows a class is thin, the
            re-captured scenes are mined for just the frames that carry it.
        select_n: randomly sample this many of the matching frames (seeded by
            select_seed). Without it every matching frame is written, which for a
            300 s scene is thousands of near-duplicates.

    Returns:
        A stats dict: frames written, per-class instance counts, and drop
        counts by reason -- this is the step 9.v histogram input.
    """
    root = Path(output_root)
    buffers = ["cam_front", "cam_front_depth", "cam_front_instance_seg"]
    if include_lidar:
        buffers.append("lidar_top")
    dirs = ["image_2", "label_2", "calib"] + (["velodyne"] if include_lidar else [])

    # A non-append write means "this tree IS this run set", so clear it first. Without
    # that, re-running over a bigger previous tree leaves orphan frames: numbering
    # restarts at 000000 and overwrites the low stems, but everything above the new
    # frame count survives on disk -- absent from frame_index.json, still shipped to
    # S3. Append explicitly skips this, since extending the tree is the whole point.
    if not append:
        for d in dirs + ["velodyne"]:
            if (root / d).is_dir():
                shutil.rmtree(root / d)
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    calib = _calib_text(configs)

    # Files are named by frame_id (SCENE_CARLAFRAME, e.g. 002_231559), NOT by a
    # running %06d counter. Sequential numbering is the KITTI convention and buys
    # drop-in compatibility with its tooling, but it makes every filename opaque:
    # nothing on disk says which scene or sim tick a frame came from, so any manual
    # or VLM-driven QA pass has to join through frame_index.json to say anything
    # about a file. frame_id is already unique (scene ids are global, carla_frame is
    # monotonic within a scene), which also makes appends idempotent -- re-writing a
    # frame overwrites itself instead of landing again under a fresh number.
    index_path = root / "frame_index.json"
    index = json.loads(index_path.read_text()) if (append and index_path.is_file()) else {}

    # A frame already in the tree is never written twice. Without this, re-mining a
    # run that was already processed re-does work and re-inflates the histogram.
    # The stem IS the frame_id, so the index keys are the guard.
    already = set(index)

    frame_ids = None
    if select_class is not None:
        frame_ids = _select_frames(select_class, select_n, select_seed, run_ids, configs,
                                   scene_ids, raw_root, prefetch, max_workers, already)
        if not frame_ids:
            raise RuntimeError(
                f"no NEW frame in the selected runs contains {select_class!r} "
                f"({len(already)} frames of this tree were already processed). "
                "Capture a new run whose scenes actually produce that class.")

    kept, dropped = {}, {}
    frames = 0

    stream = iter_cycle_frames(run_ids, configs, tuple(buffers), prefetch, max_workers,
                               scene_ids, raw_root, frame_ids, already)
    for record, bufs in islice(stream, limit):
        stem = record["frame_id"]
        lines, track_ids, stats = label_frame(record, bufs, configs)
        assert len(lines) == len(track_ids)   # the alignment is load-bearing

        Image.fromarray(bufs["cam_front"]).save(root / "image_2" / f"{stem}.png")
        (root / "label_2" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (root / "calib" / f"{stem}.txt").write_text(calib)
        if include_lidar:
            # KITTI velodyne is right-handed (y LEFT); CARLA's LiDAR is y-right.
            pts = np.array(bufs["lidar_top"], dtype=np.float32)
            pts[:, 1] *= -1.0
            pts.tofile(root / "velodyne" / f"{stem}.bin")

        # ego_pose rides along with the frame index. It is not a KITTI field, but the
        # raw records it comes from are LOCAL ONLY and die with the capture instance,
        # so leaving it out means the published dataset can never reconstruct where
        # the ego was. That trajectory is what makes a stale detection scoreable
        # against what was true when it was consumed -- the measurement this whole
        # project exists to make. ~250 bytes/frame against ~2.7 MB of image.
        index[stem] = {
            "scene_id": record["scene_id"],
            "frame_id": record["frame_id"],
            "carla_frame": record["carla_frame"],
            "timestamp_sim_s": record["timestamp_sim_s"],
            "ego_pose": record["ego_pose"],
            "objects": len(lines),
            "track_ids": track_ids,     # aligned row-for-row with label_2/<stem>.txt
        }
        for k, v in stats["kept"].items():
            kept[k] = kept.get(k, 0) + v
        for k, v in stats["dropped"].items():
            dropped[k] = dropped.get(k, 0) + v
        frames += 1

    index_path.write_text(json.dumps(index, indent=1))
    summary = {
        "frames": frames,
        "frames_in_tree": len(index),
        "objects": sum(kept.values()),
        "empty_frames": sum(1 for e in index.values() if e["objects"] == 0),
        "class_histogram": dict(sorted(kept.items())),
        "dropped": dict(sorted(dropped.items())),
    }
    (root / "write_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def test():
    # scene_ids = _run_scene_ids(CONFIGS, 1)
    # print(scene_ids)

    print(CONFIGS["CARLA_config.json"]["weather_presets"])
    
    
    

if __name__ == "__main__":
    test()