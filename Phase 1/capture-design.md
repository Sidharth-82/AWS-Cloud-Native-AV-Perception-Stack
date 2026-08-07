# Capture System — Final Design (Phase 1 steps 5–7)

The GPU-side capture pipeline: drive the CARLA server, record raw frames + labels per
`config/dataset_example.json`, stream to S3 per scene. Counterpart to the offline reader
(`src/utils.py::iter_run_frames`). Runs on the `carla-0915-ready` AMI, attended, subset-first.

## Three concerns (one file each)
| File | Concern | Knows about |
|------|---------|-------------|
| `src/scene.py` — `Scene` | **Control.** One scene's world + rig lifecycle; returns raw per-frame snapshots. | CARLA only. NOT the dataset schema, NOT disk layout. |
| `src/serializer.py` | **Persist + format.** Snapshot → `records.jsonl` line + sensor files on scratch disk. | `dataset_example` schema + `naming_convention` + scratch paths. |
| `src/capture.py` — `Capture` | **Orchestrate.** Parse configs, iterate scenes, drive the tick loop + cadence + subset cap, per-scene S3 upload, once-per-run calib/provenance. | Run flow, S3, provenance. |
| `src/utils.py` | Shared config loaders (`load_configs`, `strip_all_documentation`) + the offline reader. | (existing) |

Reuse from `utils.py`: `load_configs()`, `_run_scene_ids()`, `_scene_prefixes()`, and the
`naming_convention` format strings from `CARLA_config.json` — so writer keys == reader keys.

## `Scene` lifecycle (context manager)
- **`__enter__`**: `world_setup()` inside try/except; on failure call `_cleanup()` then re-raise
  (preserve the original cause — `raise SetupError from e`, not a bare new Exception, so the real
  error isn't masked). Returns `self`.
- **`world_setup()`**: load map; save original world settings; set **synchronous_mode + fixed
  delta**; TM sync + seeds; spawn ego (`vehicle.tesla.model3`) + N traffic (density preset,
  exclude bicycle); attach the `ego_config` rig; **register `sensor.listen()` once here** (each
  callback enqueues `(data.frame, data)` onto a per-sensor thread-safe queue); create **fresh
  queues per scene**.
- **`start()` / `stop()`**: autopilot **on / off** only ("freeze" marker before final flush; in
  sync mode nothing moves once ticking stops anyway).
- **`tick()`**: `frame = world.tick()`; drain each sensor queue for the payload matching `frame`
  (match on `carla_frame` — never assume "latest callback"). Returns the frame's raw sensor set.
- **`get_actors()` / `get_signs()`**: vehicles from the actor registry; signs from
  `get_environment_objects(TrafficSigns)` + `get_all_landmarks_of_type('274')` associated by
  proximity → `listed_speed_kph`. Both **unfiltered** (visibility is offline step 8).
- **`snapshot()`**: assemble the raw per-frame struct (below).
- **`_cleanup()`**: `sensor.stop()` → destroy sensors → destroy vehicles (`apply_batch(DestroyActor)`)
  → **restore original world settings (revert sync mode)**. Called by BOTH `__exit__` and failed
  `__enter__`. **No `__del__`** (non-deterministic, may never run, swallows exceptions).
- Scope: `Scene` = one scene. The **client connection is an outer lifetime** (connect once, reuse).

## Snapshot interface (`Scene` → serializer)
Raw struct, no schema/disk knowledge:
`carla_frame`, `frame_id`, `timestamp_sim_s`, `ego_pose`, `actors[]` (id, transform, world bbox,
blueprint_id, base_type, velocity), `signs[]` (world box, listed_speed_kph), `sensor_payloads{}`
(raw CARLA buffers keyed by sensor name — **in memory**, passed by reference).

Buffer sizing is a non-issue: ~11 MB/frame, returned by reference (no copy). Memory stays flat
because the loop serializes and releases **one frame at a time**; the per-scene buffer holds only
**file paths + record lines**, never pixels. The uploader works off paths, not memory.

## Serializer
Per snapshot: write sensor files to scratch, append one `records.jsonl` line, return paths.
- **Depth + instance-seg: RAW lossless PNG** via `image.save_to_disk(path, ColorConverter.Raw)` —
  no converter (`CARLA_config.sensor_encoding._CRITICAL_lossless`). RGB lossless PNG. LiDAR `.npy`
  float32 (N,4). `sensor_files` paths relative to the `raw/` root (start with `scene_XXX/`).

## Capture (orchestration) + the tick loop
Sync mode ⇒ **no real-time pressure** (server blocks on your `tick()`), so: **sequential loop,
synchronous serialization, no task-manager/scheduler.** The only concurrency is a **background
S3 uploader** (ThreadPoolExecutor, mirrors the reader's prefetch pool).

Per run: stamp calib (from `ego_config`) + `metadata.json` provenance **once** (git commit or
config-hash-only, `carla_version_observed`, python, ami, region, timing, cost). Then per scene:
```
with Scene(scene_cfg) as scene:
    scene.start()
    for i in range(frames_this_scene):        # subset: --max-frames-per-scene caps this
        for _ in range(capture_every_n_ticks): scene.tick()   # 10 ticks -> 2 Hz
        serializer.write(scene.snapshot())     # writes files to nvme, records line
        maybe flush a chunk to S3 (background)
    scene.stop()
upload scene prefix -> verify -> delete local -> mark done
```
Idempotent: re-running a scene overwrites its prefix; skip scenes whose prefix already exists
(resumable after a spot kill).

## S3 layout (must match `utils._iter_fetch_jobs`)
```
s3://perception-project-bucket/raw/scene_001/records.jsonl
s3://perception-project-bucket/raw/scene_001/cam_front/001_000123.png
s3://perception-project-bucket/raw/scene_001/{cam_front_depth,cam_front_instance_seg}/001_000123.png
s3://perception-project-bucket/raw/scene_001/lidar_top/001_000123.npy
```

## Spot resilience
Spot g4dn interruptions are uncommon but real. **Both nvme and the delete-on-termination EBS root
die on a spot kill** — so disk choice is NOT the protection; **S3 flush cadence is.** Keep nvme
scratch (fast, free); flush per scene (or per-N-frames for tighter bounds). Best mitigation: poll
the interruption notice at IMDS `/latest/meta-data/spot/instance-action`; on the ~2-min warning,
break, flush the current partial scene, then die.

## Config / infra fixes (do first)
- **Bucket name**: `scene_description.json` (`scenes[].storage.s3_prefix`) + `metadata.json`
  (`storage_root`): `carla-perception-v1` → **`perception-project-bucket`**. Uploader + reader read
  these verbatim.
- **`Dockerfile`**: add **`boto3`** to the pip line (uploads need it; currently absent).
- `capture.py` imports `utils` — run as a package or fix `sys.path` so the import resolves on the box.

## Deployment (attended, subset-first)
Two containers on **`--net=host`** (client reaches `localhost:2000` AND the IMDS role creds).
`scp src/ config/ Dockerfile orchestrate.sh` → build `carla-client:0.9.15` once → `orchestrate.sh`
starts the server container, waits for `:2000`, runs the client with `-v ~/cap:/workspace`.
First: `--max-frames-per-scene 5`, inspect, then full run. Terminate only after verifying S3.

## Verification (cheap → full)
1. **Local, no GPU**: configs load/strip, run 1 → scene_ids, scene→prefix uses the real bucket.
2. **Subset run** (`--max-frames-per-scene 5`): S3 has `records.jsonl` + 4 buffers for all 12 scenes.
3. **Writer↔reader contract**: `utils.iter_run_frames(1)` over the subset decodes depth +
   instance-seg and yields `(record, buffers)` without error (resolves the `utils.py:61` TODO).
4. **Empirical checks** before the full run: instance-seg id decodes to `actor.id` + byte order
   (`sensor_encoding.how_to_verify`); enumerate per-town sign values vs `class_map`.

## Open items (not this module)
- `utils._decode_*` and `_iter_fetch_jobs/iter_run_frames` are still `pass` stubs — the offline
  half must be filled before verification step 3 fully passes (separate task).
- No git repo yet (`metadata.provenance.git_commit`): `git init` or record config-hash-only.
- CARLA→KITTI 3D transform is offline (step 8); this writer only emits world-frame boxes per
  `coordinate_conventions` — keep them exact, they're the transform's input.
