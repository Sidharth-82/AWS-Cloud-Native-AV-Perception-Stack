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

## On-disk layout (raw)

Written to the instance store, read by the processing container off local disk. Raw is
**not** published: capture and processing share a box, so uploading it would mean ~200k
PUTs for data the next stage reads locally. Only the finished dataset goes to S3.
```
/opt/dlami/nvme/perception/_scratch/scene_001/records.jsonl
/opt/dlami/nvme/perception/_scratch/scene_001/cam_front/001_000123.png
/opt/dlami/nvme/perception/_scratch/scene_001/{cam_front_depth,cam_front_instance_seg}/001_000123.png
/opt/dlami/nvme/perception/_scratch/scene_001/lidar_top/001_000123.npy
```
`--upload-raw` still pushes raw under `run_XXX/scene_XXX/...` for a run that must outlive
its instance; the offline reader takes either source behind one interface.

## Spot resilience
Spot g4dn interruptions are uncommon but real, and **both nvme and the delete-on-termination
EBS root die on a spot kill** — so disk choice is not the protection. The protections that
matter: capture is **per-run**, so a kill costs the current run rather than the set, and
completed runs stay on disk for the processing pass. Poll the interruption notice at IMDS
`/latest/meta-data/spot/instance-action` and use the ~2-min warning to finish the current
scene. Raw is disposable by design; the dataset in S3 is what must survive.

## Night lighting
CARLA's autopilot drives with lights off and `clear_night` puts the sun at -90 degrees, so a
night scene has no light source at all. Every vehicle in a night scene gets `Position | LowBeam`
set explicitly rather than through the Traffic Manager, which would make a scene's exposure
depend on TM state instead of on the scene config.

## Per-scene capture rate
`capture_every_n_ticks` defaults from `CARLA_config` but a scene may override it. Instances of a
scarce class scale as `density x duration x range x capture_hz` — ego speed cancels, since
driving slower holds an object in frame longer but passes proportionally fewer. Rate is
therefore the only per-scene lever that multiplies a starved class.

## Deployment
`orchestrate.sh` runs the whole pipeline: start the server, install the additional maps if
missing, build **both** images, capture every run, process every split, publish. Two images
because the capture client is pinned to Python 3.7 by the `carla` wheel while the offline half
is numpy on 3.11; neither copies source, both mount the checkout, so a code edit needs no
rebuild. Both run `--net=host` so the client reaches `localhost:2000` and the IMDS role creds.
Subset first (`FRAMES=5`), inspect the renders, then the full run.

## Verification (cheap -> full)
1. **Local, no GPU**: configs load/strip; runs resolve to scene ids; every run is split-pure and
   agrees with the per-scene split.
2. **Subset run** (`bash orchestrate.sh`): capture, processing, splits and the S3 layout end to
   end across every scene, for a few minutes of GPU.
3. **Writer/reader contract**: stream the subset back; depth and instance-seg decode without error.
4. **Convention checks**: depth planar-vs-radial and LiDAR handedness resolved against each other
   with LiDAR reprojection; `rotation_y` checked against recorded velocity.
5. **Render the cuboids**: `viz.py` rebuilds boxes from the written label fields and draws them,
   which round-trips the writer. A wireframe box is symmetric, so the facing end is drawn with a
   cross — otherwise a 180-degree yaw error renders perfectly.
