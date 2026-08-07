# CARLA on EC2 — Relaunch Runbook

Baked AMI so the CARLA box comes back in ~2 minutes instead of redoing driver/Docker/pull setup.

## The AMI
- **Name:** `carla-0915-ready`  (AMI id: `ami-________` — fill in from EC2 → AMIs)
- **Contents:** Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) + `carlasim/carla:0.9.15` pre-pulled into `/var/lib/docker` (on the EBS root, so it's captured in the image).
- **Root volume:** 120 GB gp3 (DLAMI base ~49 GB + CARLA image + headroom).
- GPU-validated: server boots headless on the T4 and listens on port 2000.

## Fixed facts for this project
| Thing | Value |
|---|---|
| Region | `us-east-1` |
| Instance type | `g4dn.xlarge` (Spot) |
| Instance profile (role) | `EC2-S3-Role` — gives the box S3 read/write, no keys on disk |
| S3 bucket | `perception-project-bucket` |
| SSH key | `.ssh\carla-phase1.pem`|
| Security group | SSH (22) from my IP only; ports 2000-2002 stay local (no inbound rule) |

## Relaunch steps

### 1. Launch from the AMI
EC2 → Launch instance → **My AMIs** → `carla-0915-ready`. Set:
- Instance type `g4dn.xlarge`
- Purchasing option: **Spot**, one-time
- Key pair: `carla-phase1`
- Security group: the existing SSH-from-my-IP one
- **Advanced → IAM instance profile: `EC2-S3-Role`**  ← easy to forget; without it the box can't reach S3
- Root volume stays 120 GB gp3

> Note: my public IP changes, so update the security group's SSH source rule if SSH times out.

### 2. Connect
```powershell
ssh -i "\.ssh\carla-phase1.pem" ubuntu@<PUBLIC_IP>
```

### 3. Start CARLA (image is already there)
```bash
docker run -d --name carla --gpus all --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound
```

### 4. Quick sanity check
```bash
docker ps                         # STATUS = Up
docker logs --tail 30 carla       # UE4 4.26 boot, no fatal GPU/Vulkan errors
ss -tln | grep 2000               # RPC port open on the host
```
Port 2000 open = server live and accepting client connections.

### 5. Confirm the S3 role is attached (temporary creds, not user keys)
```bash
aws sts get-caller-identity       # ARN should read assumed-role/EC2-S3-Role/...
```

## When done for the session
Spot one-time instances can't be stopped, only terminated — so terminate to stop billing:
```powershell
& "C:\Program Files\Amazon\AWSCLIV2\aws.exe" ec2 terminate-instances --instance-ids <INSTANCE_ID>
```
The EBS root deletes with it; the AMI + its snapshot persist for next time (~$3/mo).

## Gotchas learned the hard way
- **The DLAMI eats ~49 GB of the root by itself** — an 80 GB root can't unpack CARLA (needs ~40 GB peak). Hence 120 GB.
- **Instance-store (`/opt/dlami/nvme`) is NOT captured in an AMI.** Docker must stay on the EBS root (`/var/lib/docker`) for the image to bake in. Confirm with `docker info | grep "Docker Root Dir"` → `/var/lib/docker`.
- **The `carla` Python module doesn't import inside the server container** — the image ships eggs for py2.7/py3.7 but the container's Python is 3.6 (ABI mismatch), and the client also needs `libjpeg.so.8`. Run the real client from a matched **Python 3.7** env or `pip install carla==0.9.15` in a separate env — not inside the server container. Server validation only needs the port-2000 check above.
- **CLI can't create the AMI or modify volumes** — `Laptop-User` lacks `ec2:CreateImage`/`ec2:ModifyVolume` by design; do those in the console as admin. It *can* run/terminate/describe instances.

## Re-baking the AMI (if you change the image/setup)
```bash
docker stop carla && docker rm carla    # bake the image, not a running container
docker ps -a                            # empty
docker images                           # carlasim/carla:0.9.15 present
```
Then console → Instances → Actions → Image and templates → Create image (leave "Reboot" enabled). Don't use `docker system prune -a` (it would delete the CARLA image).
