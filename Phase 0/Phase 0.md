### Phase 0 — Frame the problem and lock the narrative
- **Objective:** decide the exact perception task and the story before writing any code.
- **Decide:** Which capability is the hero? Do you lean AV-heavy (aligned with your research) or broader robotics? What is the single "wow" moment in your demo video? Who is the imagined end user, and why would they care?
- **Done when:** you can say the one-liner and a 30-second pitch out loud, and you've written a one-paragraph "what this proves" statement.
- **Pitfall:** picking a task so broad you can never call it finished.

## Problem Statement
With AI in real world, that needs close to real-time intelligent decisions, today this poses a problem of high costs local hardware and hosting as opposed to cloud based or distributed processing which is cheaper. However the latter suffers from latency and potential connectivity challenges that will diminish the real-time response. 

## Opportunity
Building a solution that will use cloud-based or distributed processing for near real-time or non-real-time Artificial Intelligence decisions, while simoultaneously using local processing and local data for processes requireing real-time decisioning. This will balance the cost vs speed connondrum faced by most decisioning systems.

## Objective
A measurement study of whether cloud-hosted, near-real-time perception outputs remain usable despite their delay. The perception system runs at full strength, split into two tiers: onboard real-time (lane geometry, ≤100 ms) and cloud near-real-time (speed-limit signs, vehicle detection→tracking→lead-distance, ≤5 s). Holding strength fixed, the study measures each cloud output's end-to-end age and the error that delay introduces — by the time a result returns, has the scene changed enough that it's stale or misleading? The outcome states which perception outputs can honestly live in the cloud tier and which must stay onboard. Degrading the model for speed is a later decision these results inform.

This is a highway perception system that outputs the scene a driving policy would need: lane geometry, surrounding vehicles (tracked, with distance to the lead car), and speed-limit signs. The information gained from the perception system is the goal, to be used in control system later on (Phase 5).

Setting: Highway \
Vehicle: Sedan - Car will have cameras, lidar.

These are the following use cases that will be experimented with: 

Real-Time:
  1. Lane detection (lane geometry)

Near Real-Time
  2. Traffic-sign detection + speed-limit classification
  3. Vehicle detection → tracking → lead-vehicle distance estimation

Real-Time is defined by response required within 100 ms. 

Near Real-Time is defined by response required within 5 seconds

## The Narrative

**One-Liner**
A highway perception stack split across an onboard real-time tier and an AWS near-real-time tier, used to measure whether delayed cloud perception is still honest enough to act on.

**Pitch**
Built in CARLA, served from AWS. A simulated sedan runs lane detection onboard in real time and streams camera + LiDAR to a full-strength cloud model that detects/tracks vehicles, estimates lead-car distance, and reads speed-limit signs near-real-time. Because the cloud answer arrives seconds after the frame it was computed on, the world has moved — so I measure the delay-induced error (stale output vs. ground truth at the moment it's consumed) and find which outputs stay usable and which go stale. That's the question a perception team has to answer before putting anything in the cloud.

**What it proves**
That I can build a strong perception system *and* ask the systems question that actually gates deployment — is a delayed output still safe to act on? — by instrumenting it and measuring temporal validity, not just training accuracy.

**Imagined End-User**
An AV/robotics perception + systems team screening entry-level candidates. They care because most portfolios stop at a trained model; this one measures the deployed system's temporal validity and defends which outputs belong in the cloud — the reasoning that separates hireable candidates at this level.

**How to show**
Live overlay of tracked vehicles with distance labels + the detected speed limit, as the car drives. 

## Data to Collect

Two distinct datasets — do not conflate them.

**1. Perception data** (to train/evaluate the models — CARLA sim, KITTI format)
- Camera RGB frames, LiDAR point clouds, ego pose + velocity, timestamps
- Per-frame labels: vehicle 2D/3D boxes + track IDs, sign boxes + speed-limit class, lane geometry, ground-truth distance to lead vehicle
- A **fully timestamped ground-truth timeline**, so a stale cloud output can be scored against what was true *when it was consumed*, not when it was computed. This is what makes delay-induced error measurable.

**2. Measurement telemetry** (the actual output of the study — logged per inference request)
- **Staged timestamps:** capture → encode → tx up → inference start/end → tx down → consumed → end-to-end age (p50/p95/p99)
- **Delay-induced error, per output:**
  - Lead distance: `|distance_reported(from frame t) − distance_gt(at consume time)|`, tracked against relative velocity
  - Speed-limit sign: is the returned sign still the applicable one at consume time? (valid / stale boolean)
  - Vehicle tracks: IoU / center drift of stale boxes vs. current ground-truth boxes; ID consistency
- **Scene-dynamics covariates:** ego speed, relative speeds, sign spacing (usability depends on these)
- **Cost / bandwidth:** instance $/hr, $ per 1,000 frames, egress cost, bytes/frame sent (raw vs. compressed)
- **Throughput / resource:** sustained FPS, GPU/CPU util, memory
- Each logged point carries (accuracy, age, delay-induced error, cost) so the usability curve is plottable
- **Fixed this phase** (not swept): model strength, input resolution, instance type, network condition

## Scope Pitfalls
Scope Creep from a driving-behaviour framing. Perception ends after the 3 perception outputs completed. 

Completed defined by quantitative values. Baseline accuracy targets met (>= X mAP/IoU, distance measure error <= X meter, MOTA/IDF1), **and** for each cloud output: end-to-end age (p50/p95) logged and delay-induced error quantified vs. relative speed, with a stated usability threshold and the % of frames the delayed output stays usable. Conclusion states which outputs belong in the cloud tier vs. onboard.

