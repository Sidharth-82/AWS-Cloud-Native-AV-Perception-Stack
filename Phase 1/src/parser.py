
#################################
"""
This file is meant to house code dedicated towards parsing

Functions within:

def iter_run_frames(run_id, configs, buffers=...) -> Iterator[(record, dict)]
    - Lazily streams a run's frames from S3 one at a time: yields a per-frame
      record and a dict of DECODED sensor buffers. Bounded prefetch, nothing
      held on disk. This is the entry point for offline step 8.

"""


### Imports below

from collections.abc import Generator
import io
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor


import boto3
import numpy as np
from PIL import Image


### S3 streaming below

# Maps each sensor-buffer name (the record's sensor_files keys) to its decoder.
# Depth and instance-seg are ENCODED buffers, not literal images -- see
# CARLA_config.sensor_encoding. RGB and lidar decode literally.
_DEPTH_FAR_M = 1000  # CARLA depth far plane (sensor_encoding.cam_front_depth)


def _decode_rgb(raw: bytes) -> np.ndarray:
    """Front RGB: literal PNG -> HxWx3 uint8."""
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _decode_depth(raw: bytes) -> np.ndarray:
    """
    CARLA depth: 24-bit depth packed in RGB -> HxW float32 metres.

    depth_m = 1000 * (R + G*256 + B*256^2) / (256^3 - 1)   [CARLA_config]
    """
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    return 1000 * (R + G*256 + B*256^2) / (256^3 - 1)


def _decode_instance_seg(raw: bytes) -> np.ndarray:
    """
    CARLA instance-seg -> HxW uint32 actor id.

    id = G*256 + B   [CARLA_config]. R channel is the semantic tag (dropped).
    BYTE ORDER UNVERIFIED: if the empirical check (CARLA_config.how_to_verify)
    shows B*256 + G
    """
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    return R, G*256 + B


def _decode_lidar(raw: bytes) -> np.ndarray:
    """LiDAR: .npy float32 (N, 4) = x, y, z, intensity."""
    return np.load(io.BytesIO(raw))


_DECODERS = {
    "cam_front": _decode_rgb,
    "cam_front_depth": _decode_depth,
    "cam_front_instance_seg": _decode_instance_seg,
    "lidar_top": _decode_lidar,
}


def _run_scene_ids(configs: dict, run_id) -> list:
    """Resolve run_id -> its scene_ids via metadata.json."""
    for run in configs["metadata.json"]["runs"]:
        if run["run_id"] == run_id:
            return run["scene_ids"]
    raise KeyError(f"run_id {run_id!r} not found in metadata.json")


def _scene_prefixes(configs: dict) -> dict:
    """{scene_id: s3_prefix} from scene_description.json."""
    return {
        s["scene_id"]: s["storage"]["s3_prefix"]
        for s in configs["scene_description.json"]["scenes"]
    }


def _iter_fetch_jobs(s3, scene_ids: list, prefix_by_scene: dict):
    """
    Walk a run's scenes and yield one fetch job per frame.

    For each scene: ONE GET pulls the records file (JSONL), then a job is
    emitted per record. A job is (bucket, raw_key_prefix, record) -- everything
    the worker needs to build the per-buffer S3 keys.

    Key resolution: sensor_files paths are relative to the 'raw/' root, NOT the
    scene prefix (they already start with 'scene_XXX/'). raw_key_prefix is the
    scene prefix with its trailing 'scene_XXX/' stripped, so
    raw_key_prefix + sensor_files[name] is the correct object key.
    """
    pass


def iter_run_frames(
    run_id,
    configs: dict = CONFIGS,
    buffers: tuple = ("cam_front_instance_seg", "cam_front_depth"),
    prefetch: int = 64,
    max_workers: int = 16,
) -> Generator[tuple[dict, dict[str, np.ndarray]]]:
    """
    Lazily stream a run's frames from S3, one at a time.

    Yields (record, buffers) per frame, where record is the parsed per-frame
    dict (schema = dataset_example.json) and buffers maps each requested buffer
    name to its DECODED numpy array. Encoded depth/instance-seg are decoded
    before yielding -- raw bytes never surface.

    run_id is FIXED for the life of the generator: one generator walks one run's
    frames (scene by scene, in scene_ids order) then stops. It cannot be handed
    a new run_id mid-iteration. To span runs, chain independent generators:
        itertools.chain.from_iterable(iter_run_frames(r) for r in run_ids)

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
        max_workers: number of S3 fetch threads.

    Yields:
        (record: dict, buffers: dict[str, np.ndarray])
    """
    pass


def test():
    # scene_ids = _run_scene_ids(CONFIGS, 1)
    # print(scene_ids)
    
    # scene_prefixes = _scene_prefixes(CONFIGS)
    # print(scene_prefixes)
    
    print(CONFIGS["CARLA_config.json"]["weather_presets"])
    
    
    

if __name__ == "__main__":
    test()