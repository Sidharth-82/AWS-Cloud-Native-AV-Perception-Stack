### Phase 1 — Stand up the simulator and generate data
- **Objective:** get CARLA running and produce a labeled dataset from its sensors.
- **Decide:** What sensor suite (cameras only, or camera + LiDAR)? How much scenario diversity — weather, maps, traffic density, time of day — and how do you keep it balanced? How much data is enough? Do you pull ground-truth labels directly from the sim, or treat sim output as unlabeled? How do you split train/val/test so scenarios don't leak across splits? What on-disk format — and is it worth mirroring a known schema (e.g. nuScenes-like) so it feels familiar to reviewers?
- **Done when:** a reproducible script turns *N* scenarios into a stored, labeled dataset with a short data card describing it.
- **Pitfall:** a dataset that's too uniform (model looks great, generalizes to nothing); ignoring the sim-to-real gap in how you later *talk* about results.

# CARLA

## CARLA Settings

**Vehicle Sensor Suite:**
  1. LiDAR
  2. Front View Camera (Rig designed to add 5 more later)
  3. Vehicle Localization Sensors (GPS, Accelerometer, Wheel Encoders, Etc) - For Tracking

**Scenario Diversity:**
  1. Day/Night
  2. Clear Weather
  3. Traffic Density - Vary between Low and High to prevent model overfit
  4. Speed limit signs must vary. (90 - 110 kph)
  5. Map Variety - Long straight stretch, Curving road, Highway loop, bad lane line conditions

Add Ons once complete:
  1. Dense Traffic + Low Speed
  2. Low intensity rain, less dense snow


**How much data is enough**
Sample at low frequency ~ 2 Hz
At highway speeds, Consecutive 20 Hz frames are near identical

Vehicle instances will pile up quicker then sign instances. Target instance counter per class. 
Target between 500 - 1500 instances per class. (Classes: Vehicle, Speed Sign)

10 - 20 Runs x a few minutes per run at 2 Hz will result in around 10k frames. 

Look at per-class histogram and add class specific images to top up. 

**Where to pull ground truth labels from?**
Ground Truth Labels to be pulled directly from the sim. 
CARLA provides pre-labeled bounding boxes for all actors

Vehicles - 3D Bounding box in world coords (actor.bounding_box)

Signs - Available in Map Landmarks (map.get_all_landmarks_of_type(...)). Carries speed value, bounding box, and class label

Must ensure to filter for visible, in-frustum actors. Can be accomplished using instance-segmentation camera hit test. Important to prevent model from detecting invisible cars.

**How do you split train/val/test so scenarios don't leak?**
Split at the run/scenario level, never the frame level. 
All frames from a recorded drive go to one split. re-run drive with different scenarios to obtain data for seperate split. Shows that the model isnt just fitting to known conditions. 

Val and Test should have different conditions(change map, time, weather, speed, etc) then Train set.

Make sure to record split key in data card to ensure reproduciblity and auditability.


**On Disk Format**

If just front cam + lidar:
 - name as (image + label.txt) - KITTI Style
 - one box per line
 - 2D + 3D + class 
 - calibration file

If LiDAR + 6 Camera setup:
 - a Per-sample folder of synchronized sensor files + a JSON/parquet annotation table (sample token, sensor, boxes, ego_pose).
 - Much heavier. Will implement in another revision of this project. 


## Getting CARLA Running

CARLA is a UE4 renderer - Even without visual display, the camera and liDAR are GPU-rendered, so there's no chance to run locally without NVIDIA GPU

CARLA can be run on AWS EC2 spot GPU instance (g4dn.xlarge), a T4 16GB, which is plenty for CARLA.
Ballpark Cost: 
 - On-demand ~$0.53/Hr
 - Spot ~$0.15-0.20/Hr

Data gen will take a few hours total, so I'm expecting high single-digit dollars per data collection cycle. 

**Cost Minimizing Architecture**
Split the pipline so that the GPU is only up for the part that truly needs it

On the GPU instance: Run CARLA, drive the scenarios, and for each frame at 2 Hz save the raw outputs + Metadata records.
 - Camera images + LiDAR Points
 - Actor Bounding boxes + transform
 - Sensor calibration
 - Ego vehicle position
 - Weather/map/time tags
Dump all this info to an S3 instance. 

Once finished terminate the instance. Might be slower to perform labeling locally, but it is free.

Offline / Locally: Pull the raw capture and do all the CPU Work, project the actor boxes into 2D/3D, run the occlusion/in-frustom filter, attach the sign labels, package into KITTI format, assign splits, and write the data card. 

### Steps to take

**A. AWS + CARLA setup**

1. Set a budget alarm and Request G-instance quota increase

    i. Create a non-root IAM user with least-privilege for EC2 + S3; configure AWS CLI.

    ii. AWS Budgets → monthly cost budget (e.g. $20) with alerts at 50/80/100%.

    iii. Service Quotas → EC2 → "All G and VT **Spot** Instance Requests" → request ≥4 vCPUs (g4dn.xlarge = 4). Do this first — approval can take up to a day.

    iv. Pick one region; create the S3 bucket in that same region (avoids cross-region egress cost).

2. Launch g4dn.xlarge spot with a Ubuntu + carlasim/carla Docker Image with nvidia-container-toolkit

    i. Launch g4dn.xlarge as a spot request; Ubuntu 22.04 (or a Deep Learning AMI with NVIDIA drivers preinstalled).

    ii. If not a DLAMI: install NVIDIA driver + Docker + nvidia-container-toolkit; verify nvidia-smi inside a container.

    iii. docker pull carlasim/carla:0.9.15 (pin the version).

    iv. Attach an instance IAM role granting S3 write (cleaner than putting keys on the box).

    v. Security group: SSH from your IP only; CARLA ports (2000-2002) stay local to the instance.

3. Run CARLA server headless (~RenderOffScreen); confirm with a test client. Pin the CARLA version and a matching carla Python Wheel in a dedicated venv. Wheel lags new Python versions, Check supported python instances

    i. docker run --gpus all ... carla ... -RenderOffScreen (headless render).

    ii. Create a venv/conda env; pip install carla==<server version> + numpy/opencv/pyyaml. Verify the supported Python version first (the wheel lags).

    iii. Smoke test: connect client → get world → list maps → spawn one vehicle → grab one camera frame. Confirms the full loop before you invest GPU hours.

4. CARLA must be run in synchronous mode with a fixed timestep and tick the world manually.

    i. Set settings.synchronous_mode=True and fixed_delta_seconds=0.05 (20 Hz sim step); tick manually and capture every 10th tick for 2 Hz.

    ii. Set the Traffic Manager to sync too: tm.set_synchronous_mode(True).

    iii. Fix all seeds (Traffic Manager + spawn RNG) so runs are reproducible.

**B. Data Gen pipeline**

5. Write Scenario config (map, weather, time-of-day, traffic density, speed, duration) script for CARLA to ensure re-producibility

    i. Define a config schema (YAML): map, weather preset, time-of-day, traffic density, ego target speed, route/spawn point, duration, seed.

    ii. Enumerate the scenario matrix (map × day/night × density × sign-value set) → the concrete list of N runs.

    iii. Config loader that applies map, weather, and seeds to the world.

    iv. Each run writes a manifest (exact config + git commit) for reproducibility.

6. Spawn ego vehicle + traffic via CARLA Traffic Manager, attach front camera + LiDAR + ego-state sensors. Ego Vehicle needs set_autopilot(True). Allows vehicle to move autonomously for reproducible continous data gathering.

    i. Spawn ego; set_autopilot(True) via TM at the target speed/lane behavior.

    ii. Spawn N traffic vehicles at spawn points with autopilot; density per config.

    iii. Attach sensors: front RGB (resolution, FOV, transform), LiDAR (channels, range, rotation freq, points/sec), instance-seg camera (for the visibility filter), ego-state (IMU, GNSS).

    iv. Record calibration once per run: camera intrinsics + all sensor extrinsics/transforms.

    v. Register sensor listen callbacks that buffer each tick's payload keyed by frame id (so sync-mode data stays aligned).

7. Capture Raw frames at 2 Hz + metadata and send to S3

    i. Main loop: world.tick(); every 10th tick, collect the synchronized sensor payloads.

    ii. Per captured frame, serialize: camera PNG, LiDAR .npy, instance-seg PNG, and a metadata JSON = all vehicle bboxes+transforms, ego pose, weather/map/time tags, calibration, frame id + timestamp.

    iii. Stream each run/chunk to S3 as you go (aws s3 sync), verify upload, delete local — so a spot interruption loses at most the current chunk.

    iv. Stop/terminate the instance when the run set completes.

**C. Offline Processing**

8. Project Boxes, run the occlusion filter, pull sign landmarks and values

    i. Pull a run from S3.

    ii. Per frame: transform vehicle 3D bboxes world→camera, project to 2D via intrinsics; also emit the 3D box in camera/LiDAR frame.

    iii. Visibility filter (instance-seg): keep an actor only if ≥T pixels of its ID land inside its projected box, it's in front of the camera, and within image bounds.

    iv. Signs: query map landmarks near the ego along the route, project to camera, keep in-frustum, label class = posted speed value.

    v. Sanity-visualize: render 3D boxes onto ~10 frames and eyeball them before mass-processing. KITTI 3D coordinate system is different the CARLA. Use frames to accomodate. KITTI -> (x-right, y-down, z-forward), CARLA -> (x-forward, y-down, z-up)

9. Package to KITTI, assign splits by run/map, generate the per-class instance histogram.

    i. Write KITTI layout: image_2/, velodyne/, label_2/, calib/.

    ii. Per-frame label.txt: one line per object — class, 2D bbox, 3D dims/location/rotation_y (occlusion/truncation approximated).

    iii. calib file (P2, Tr_velo_to_cam) from your recorded calibration.

    iv. Assign split by run/map per your split key; assert no run spans two splits.

    v. Compute the per-class instance histogram per split; flag under-target classes (signs) → schedule top-up runs.

10. Write the data card (scenarios, counts, per-class histogram and split key)

    i. Document the scenario matrix, #runs, #frames, per-class instance counts (train/val/test), and map/weather/time coverage.

    ii. Split-key table (run → split; which conditions are held out for val/test).

    iii. Sensor spec + calibration summary.

    iv. Reproducibility block: CARLA version, seeds, config manifests, commit hash, exact regenerate command.

    v. The sim-to-real disclaimer (sim-labeled; real-world performance untested).