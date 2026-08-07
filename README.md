# Cloud-Native AV Perception Stack

**A highway perception stack split across an onboard real-time tier and an AWS near-real-time tier, built to measure whether delayed cloud perception is still safe to act on.**

![Status](https://img.shields.io/badge/status-Phase%201%20of%207-orange)
![Python](https://img.shields.io/badge/python-3.7%20client%20%2F%203.10%20offline-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.15-informational)
![AWS](https://img.shields.io/badge/AWS-EC2%20spot%20%2B%20S3-232F3E)
![Data](https://img.shields.io/badge/dataset-KITTI%20format-lightgrey)

<!-- HERO MEDIA TODO: add docs/media/hero.gif (front camera + projected 3D boxes,
     or the Phase 5 live-overlay clip) and embed it here. Left out deliberately
     rather than committing a broken image link. -->

A simulated sedan drives a highway in CARLA. The perception that interprets what it sees is deliberately split in two: **lane geometry runs onboard** under a real-time budget (<100 ms), while **vehicle detection, tracking, lead-vehicle distance, and speed-limit sign reading run in the cloud** under a near-real-time budget (<5 s).

The models are not the point. The point is that a cloud answer describes a moment that has already passed, and this project measures how much that costs.

---

## Contents

- [The question](#the-question)
- [Status](#status)
- [Architecture](#architecture)
- [The dataset (data card)](#the-dataset-data-card)
- [Reproducing it](#reproducing-it)
- [Repository layout](#repository-layout)
- [Design decisions and tradeoffs](#design-decisions-and-tradeoffs)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## The question

Running perception on the vehicle means paying for a GPU in every vehicle. Running it in the cloud is far cheaper per unit of compute, but you pay in latency and connectivity risk.

Rather than picking a side, this project holds model strength fixed, runs the full-strength model in the cloud, and measures the thing that actually gates the decision: **by the time the answer arrives, the world has moved.**

At 100 km/h the ego covers about 28 m per second. A 2 second round trip means the answer that just landed describes a scene from roughly 56 m ago. So each cloud output is scored on **end-to-end age** and on the error that age introduces:

| Output | Delay-induced error metric | Why this metric |
|---|---|---|
| Lead-vehicle distance | `abs(reported(t) - ground_truth(t_consumed))` in meters, bucketed by relative velocity | Continuous quantity, and staleness scales with closing speed, not ego speed |
| Speed-limit sign | valid / stale boolean | A speed limit cannot be "slightly wrong": either the sign still applies at consume time or it does not |
| Vehicle tracks | IoU and box-center drift vs current ground truth, plus track ID consistency | Spatial error, and an ID that flips mid-wait breaks anything downstream |

Every logged inference carries `(accuracy, age, delay_induced_error, cost)`, so the usability curve is plotted rather than asserted. The deliverable is a per-output verdict on **which perception outputs can honestly live in the cloud tier and which must stay onboard**, against a usability threshold fixed in advance.

This is only measurable because the dataset records a **fully timestamped ground-truth timeline**, letting a stale output be scored against what was true when it was *consumed*, not when it was *computed*.

---

## Status

Built in phases, each with a written definition of done before any code. The phase docs live in this repo and are the design record.

| Phase | Scope | State |
|---|---|---|
| 0 | Frame the problem, lock scope and narrative | Complete |
| 1 | CARLA capture pipeline plus labeled KITTI dataset | Complete |
| 2 | Train and validate the detector locally | In progress |
| 3 | Move training to AWS, versioned and tracked | Planned |
| 4 | Serve as a real inference endpoint | Planned |
| 4b | Same ONNX model on a Raspberry Pi, benchmarked head to head | Planned |
| 5 | Close the loop, CARLA to AWS in real time, with delay telemetry | Planned |
| 6 | Monitoring, IaC, CI/CD, automated teardown | Planned |
| 7 | Demo, docs, write-up | Planned |

Full phase definitions: [Project Outline.md](./Project%20Outline.md) · [Phase 0.md](./Phase%200.md) · [Phase 1/Phase 1.md](./Phase%201/Phase%201.md)

---

## Architecture

```
  ── ONLINE PATH (Phase 5) ───────────────────────────────────────────────────
  CARLA (Python client)        onboard tier
  ├─ ego sedan + traffic  ───►  lane geometry  ──────────► <100 ms ──┐
  ├─ RGB / depth /                                                   │
  │  instance-seg cams         ingestion (HTTP | Kinesis | MQTT)     │
  ├─ 64-ch LiDAR         ───►   ─────────────┐                       │
  └─ IMU / GNSS                              ▼                       ▼
                                   AWS inference service      live overlay
                                   detect ─► track ─► dist    + telemetry
                                             │                       ▲
                                             └─ staged timestamps ────┘
                                                (capture, encode, up,
                                                 infer, down, consumed)

  ── OFFLINE PATH (Phases 1-3, built first) ──────────────────────────────────
  CARLA on EC2 g4dn spot ─► raw capture ─► S3 ─► CPU offline processing
                                                 ├─ project boxes to 2D/3D
                                                 ├─ occlusion + frustum filter
                                                 ├─ KITTI packaging + splits
                                                 └─ data card
                                                          │
                                                          ▼
                                                cloud training ─► artifact
```

The offline path is built first, because the online loop has nothing to serve without it.

**Cost-driven split.** CARLA is a GPU renderer, so the GPU instance does only what needs a GPU: drive the scenarios, dump raw sensor buffers and unfiltered labels, stream to S3, terminate. Every CPU-bound step (projection, occlusion filtering, class mapping, packaging) runs offline on a cheap box. Slower in wall-clock terms, far cheaper, and every offline decision becomes re-runnable without touching the GPU again.

---

## The dataset (data card)

`carla_highway_perception_v1`. Generated from the sim, labeled from the sim, KITTI on-disk format.

### Scenario matrix

12 scenes, 300 s each, captured at **2 Hz** from a 20 Hz simulation. Consecutive 20 Hz frames at highway speed are near-duplicates: they inflate frame count without adding information and they leak across splits.

| Split | Maps | Conditions | Scenes | Frames |
|---|---|---|---|---|
| train | Town04, Town06 | day + night, low + high density | 8 | 4,800 |
| val | Town04, Town06 | unseen routes and seeds, **mid** density (unseen in train) | 2 | 1,200 |
| test | **Town05** | held-out map, day + night | 2 | 1,200 |
| **total** | | | **12** | **~7,200** |

**Splits are assigned per scene, never per frame.** Test is an entirely held-out map, so the generalization probe is geometry the model has never seen. Per-scene config (routes, seeds, weather, density, target speed) is in [scene_description.json](./Phase%201/config/scene_description.json).

### Sensor rig

Rig `front_v1`, fixed across every scene. Ego is a fixed `vehicle.tesla.model3`: swapping the ego between runs would change camera height and mounting geometry, which is a distribution confound rather than useful diversity.

| Sensor | Spec | Role |
|---|---|---|
| `cam_front` | RGB, 1280x720, 90 deg FOV, at (1.5, 0, 1.6) m | Detector input |
| `cam_front_instance_seg` | co-located with `cam_front` | Vehicle visibility filter |
| `cam_front_depth` | co-located with `cam_front` | Sign occlusion check (signs are not actors, so they have no instance ID) |
| `lidar_top` | 64 ch, 100 m, 1.3 M pts/s, 20 Hz rotation, at (0, 0, 1.8) m | Lead-vehicle distance; rotation pinned to sim rate so exactly one sweep completes per tick |
| `imu`, `gnss` | ego origin | Ego state only, never a perception input |

Camera intrinsics are **derived** from width/height/FOV rather than stored beside them, because storing both invites drift. Full rig and extrinsic conventions: [ego_config.json](./Phase%201/config/ego_config.json).

### Labels

Ground truth comes directly from the sim: vehicle boxes from `actor.bounding_box` composed to world frame at capture time, signs from `world.get_environment_objects(TrafficSigns)` associated by proximity with `map.get_all_landmarks_of_type('274')` for the posted speed.

Capture-time records are **deliberately unfiltered**. Every actor CARLA reports is written, including occluded and out-of-frustum ones, and there is no `visible` field in the schema. Visibility is derived offline, so the filter threshold stays re-tunable without regenerating GPU-hours of data.

Class names are likewise not baked in. Each object stores `blueprint_id`, `base_type`, and `listed_speed_kph`; the KITTI writer applies a class map with a `fine` preset (7 classes) and a `coarse` escape hatch (5 classes, all vehicles collapsed). Re-mapping is a free offline re-run. See [CARLA_config.json](./Phase%201/config/CARLA_config.json) and the annotated per-frame schema in [dataset_example.json](./Phase%201/config/dataset_example.json).

### Per-class instance counts

<!-- Fill from metadata.json runs[].class_histogram after the offline pass. -->

Pending the full offline pass. Target is 500 to 1500 instances per class. Signs are the expected bottleneck, since a highway drive covers a long distance between them, and `motorcycle` is the at-risk vehicle class. The histogram decides two things: whether top-up scenes are needed, and whether the class map stays on `fine` or drops to `coarse`.

### Encoded buffers, read this before decoding

CARLA's depth and instance-segmentation cameras **do not emit literal images**. They pack values into RGB channels, and reading them as ordinary images yields garbage.

```
depth_m  = 1000 * (R + G*256 + B*256^2) / (256^3 - 1)   # 1000 m far plane
actor_id = (G * 256) + B                                 # R holds the semantic tag
```

Both **must** be stored as lossless PNG with no `ColorConverter` applied. JPEG would corrupt the packed channel values and silently break both decodes, and CARLA's depth converters are visualization helpers that discard precision.

### Provenance

Every run stamps: git commit, SHA256 of each config file as it existed at generation time, the CARLA version the server actually reported (checked against the pin, since a mismatch is a real bug), Python version and wheel, AMI, region, GPU hours, estimated cost, and any spot interruptions. Schema in [metadata.json](./Phase%201/config/metadata.json).

### Sim-to-real disclaimer

Everything here is simulator-generated and simulator-labeled. Reported results are sim results. Real-world performance is **untested**, not implied.

---

## Reproducing it

> **Cost warning.** Capture needs an NVIDIA GPU. CARLA renders camera and LiDAR on the GPU even when headless, so there is no CPU-only capture path. On `g4dn.xlarge` spot this runs roughly 0.15 to 0.20 USD/hr against 0.53 on demand, and a full matrix run lands in single-digit dollars. Set an AWS budget alarm before launching anything, and terminate when done.

### Prerequisites

- An AWS account with an EC2 G-instance **spot** quota of at least 4 vCPUs (approval can take a day)
- An S3 bucket in the same region as the instance, to avoid cross-region egress
- An EC2 instance profile granting S3 read/write, so no access keys ever land on the box (least-privilege policies in [AWS IAM/](./AWS%20IAM/))
- Docker with `nvidia-container-toolkit` on the instance

### 1. Bring up the CARLA server

A baked AMI (Deep Learning base AMI plus a pre-pulled `carlasim/carla:0.9.15`, 120 GB gp3 root) brings the box up in about two minutes instead of reinstalling drivers every session. Full runbook with the gotchas: [Phase 1/relaunch-carla.md](./Phase%201/relaunch-carla.md).

```bash
docker run -d --name carla --gpus all --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound

docker logs --tail 30 carla     # UE4 boots, no fatal GPU/Vulkan errors
ss -tln | grep 2000             # RPC port open == server accepting clients
aws sts get-caller-identity     # ARN should read assumed-role/<ec2-role>/...
```

### 2. Run capture

The client runs in its own container on `--net=host`, so it reaches both `localhost:2000` and the instance metadata endpoint for role credentials. It cannot run inside the server container (see [Known limitations](#known-limitations)).

```bash
docker build -t carla-client:0.9.15 -f "Phase 1/Dockerfile" .

# Smoke test first: 5 frames per scene, then inspect S3 before committing GPU hours.
docker run --rm --net=host -v ~/cap:/workspace carla-client:0.9.15 \
  python src/capture.py --max-frames-per-scene 5

# Full matrix
docker run --rm --net=host -v ~/cap:/workspace carla-client:0.9.15 \
  python src/capture.py
```

Capture is **idempotent and resumable**: each scene flushes to its own S3 prefix on completion, re-running a scene overwrites its prefix, and completed scenes are skipped. Module design: [Phase 1/capture-design.md](./Phase%201/capture-design.md).

### 3. Offline processing

CPU only, no GPU. Frames stream straight from S3 through a bounded prefetch pool and are decoded in-stream, so nothing is staged on local disk and memory stays flat regardless of dataset size.

```python
from src.parser import iter_run_frames

# Only the buffers you ask for are fetched; each extra buffer is another GET per frame.
for record, buffers in iter_run_frames(
    run_id=1,
    buffers=("cam_front_instance_seg", "cam_front_depth"),
):
    ...
```

### Verification ladder

Cheapest first, so failures surface before they cost money.

1. **Local, no GPU.** Configs load and strip, run 1 resolves to its scene IDs, scene prefixes resolve to the real bucket.
2. **Subset run.** `--max-frames-per-scene 5` proves the S3 layout end to end across all 12 scenes.
3. **Writer/reader contract.** Stream that subset back and confirm depth and instance-seg decode without error.
4. **Empirical decode check.** Spawn one vehicle at a known position and confirm the instance-seg byte order and that the decoded ID equals `actor.id`. Ten minutes, and it prevents a silently empty visibility filter.
5. **Sign enumeration.** List the distinct speed values each town actually defines before locking the class list.

---

## Repository layout

```
.
├── Project Outline.md             # the seven-phase design brief and decision framework
├── Phase 0.md                     # problem framing, scope locks, the narrative
├── Phase 1/
│   ├── Phase 1.md                 # phase plan: decisions, definition of done, steps
│   ├── capture-design.md          # capture module design: concerns, lifecycle, S3 layout
│   ├── relaunch-carla.md          # EC2/AMI runbook plus gotchas learned the hard way
│   ├── config/
│   │   ├── CARLA_config.json      # how the sim runs: server, conventions, encodings, presets
│   │   ├── ego_config.json        # what the ego is: blueprint + sensor rig (static)
│   │   ├── scene_description.json # what to capture: 12-scene matrix + split policy
│   │   ├── metadata.json          # run provenance, cost, results, class histogram
│   │   └── dataset_example.json   # the per-frame record schema, annotated
│   ├── src/
│   │   ├── scene.py               # CONTROL: one scene's world + rig lifecycle
│   │   ├── serializer.py          # PERSIST: snapshot -> records.jsonl + sensor files
│   │   ├── capture.py             # ORCHESTRATE: tick loop, cadence, S3 upload, provenance
│   │   ├── parser.py              # OFFLINE: S3 streaming reader + buffer decoders
│   │   └── utils.py               # config loading, recursive doc-key stripping
│   └── Dockerfile                 # Python 3.7 client image (matches the CARLA wheel)
└── AWS IAM/                       # least-privilege policies for the user and the EC2 role
```

Raw sensor data lives in S3 and is never committed.

### Configuration is a contract

Four config files with strictly separated concerns: how the sim runs, what the ego is, what to capture, and what actually happened. Two rules make them work.

- **Any key starting with `_` is documentation, not data**, and the loader strips them recursively, including inside arrays. Rationale lives beside the value it explains instead of rotting in a separate doc. The recursion is a hard requirement, not a nicety: notes are nested inside iterable containers, so a shallow loader would hand a prose sentence to `open()` as a file path.
- **Every ambiguity is written down**, because each one is otherwise a silent wrong-label bug. CARLA world frame is left-handed X-forward Z-up, rotations are in degrees, bounding box extents are half-dimensions, sensor transforms are `T_sensor_to_ego`, and timestamps are sim time rather than wall clock. The KITTI writer converts from those conventions to KITTI's, and that conversion is only correct because both sides are stated explicitly.

---

## Design decisions and tradeoffs

| Decision | Options considered | Chosen | Why |
|---|---|---|---|
| Hero perception task | 3D detection, segmentation, BEV, scene QA | 2D vehicle detection to tracking to lead distance, plus speed-sign classification | Finishable, and it produces exactly the outputs the delay study needs to score |
| Sensor modality | camera only vs camera + LiDAR | Both captured, camera-only for the v1 model | Capture is one-shot and GPU-expensive, so record everything; keep model scope small |
| Capture rate | 20 Hz vs 2 Hz | 2 Hz | Consecutive highway frames at 20 Hz are near-duplicates that inflate count and leak across splits |
| On-disk format | nuScenes-like vs KITTI | KITTI | A single front camera fits KITTI cleanly and it is instantly familiar to reviewers |
| Split granularity | per frame vs per scene | Per scene, with a fully held-out map for test | Frame-level splits leak; an unseen map is the strongest generalization probe available |
| Label filtering | at capture vs offline | Offline | Filter thresholds stay re-tunable without re-running the GPU |
| Class granularity | fixed list vs config knob | Config knob, `fine` with a `coarse` escape hatch | Raw attributes are recorded, so class re-maps and collapses are free offline re-runs |
| Where labeling runs | GPU instance vs cheap CPU | CPU, offline | Projection and filtering are pure numpy; paying GPU rates for them is waste |
| Capture compute | local, on-demand, spot | `g4dn.xlarge` spot plus a baked AMI | Roughly a third of on-demand cost, and the AMI removes the per-session setup tax |
| Spot durability | bigger disk vs flush cadence | Flush cadence per scene, plus IMDS interruption polling | Both NVMe and the delete-on-termination root die with the instance, so disk choice protects nothing |
| Sim determinism | async vs sync fixed timestep | Sync, fixed 0.05 s, seeded TM and spawn RNG | Async gives misaligned sensor frames and non-reproducible runs |
| Frame/sensor sync | latest callback vs frame-ID match | Match on the integer from `world.tick()` | Assuming latest-wins in a multi-sensor rig is how you silently mislabel data |
| Credentials | access keys on the box vs instance role | Instance role | No long-lived secrets on disk, and credentials auto-rotate |

---

## Known limitations

Stated plainly, because these are the questions a reviewer should be asking.

- **Sim only.** No real-world data and no sim-to-real validation. Results are labeled as sim results throughout.
- **Front camera only.** The rig is built so five more cameras can be added without a schema change, but v1 is forward-facing, so nothing behind or beside the ego is perceived.
- **Weather diversity is thin in v1.** Clear day and clear night. Rain, fog, and low-sun glare presets exist in config but are add-on round, not in the must-complete matrix.
- **Sign classes are limited to what the maps define.** CARLA speed-limit signs come from each town's OpenDRIVE landmarks, and stock towns commonly only define 30/60/90 kph. The original 90 to 110 target was not achievable without spawning custom props, so the class list follows what the maps actually offer.
- **The `fine` class preset may be unaffordable.** Seven classes each wanting 500 to 1500 instances is a lot of highway driving, and `motorcycle` is the likely casualty. The `coarse` preset exists for exactly this outcome.
- **The `carla` wheel pins the client to Python 3.7**, and the module will not import inside the server container at all: the image ships eggs for 2.7 and 3.7 while the container's own Python is 3.6, and a shared library is missing. Hence the separate client container.
- **The instance-seg ID decode must be verified empirically.** Byte order is asserted in config but treated as unverified until checked against a known `actor.id`.
- **No lane-geometry labels yet.** The onboard real-time tier is specified but its dataset is not part of the Phase 1 matrix.

---

## Roadmap

Next, in order:

1. **Phase 2:** train the detector, beat a defined baseline on the held-out map, render qualitative overlays. The baseline is fixed before training starts, so "the model works" is a claim with a number behind it.
2. **Phase 3:** containerized training on AWS with experiment tracking and a versioned artifact in S3.
3. **Phases 4 and 4b:** serve it, then convert to ONNX and run the same model on a Raspberry Pi for a head-to-head cloud versus edge benchmark on identical inputs.
4. **Phase 5:** close the loop and collect the real delay telemetry, which is where the central question finally gets answered with data.
