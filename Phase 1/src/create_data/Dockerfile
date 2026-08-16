FROM python:3.7-slim

# Runtime libs: libGL/glib for opencv, jpeg/png for Pillow + carla client
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libjpeg62-turbo libpng16-16 && \
    rm -rf /var/lib/apt/lists/*

# carla wheel is py3.7-only. Pillow + numpy + boto3 are pulled in transitively
# by utils.py (PIL.Image, numpy, boto3), so they must be present for `import utils`.
RUN pip install --no-cache-dir \
      carla==0.9.15 \
      numpy==1.21.6 \
      opencv-python==4.6.0.66 \
      pyyaml==5.4.1 \
      boto3==1.33.13 \
      Pillow==9.5.0

# Unbuffered stdout/stderr so print()/tracebacks appear live and survive a hard
# crash (segfault/OOM). Without this, Docker's non-TTY pipe block-buffers Python
# output and a killed process loses it all. Placed after pip so the layer stays cached.
ENV PYTHONUNBUFFERED=1

# common/ holds config_loader + utils, shared with the offline image. Put it on
# the path rather than duplicating it, so both halves read configs through one
# loader and cannot drift on how documentation keys are stripped.
ENV PYTHONPATH=/workspace/src/common

WORKDIR /workspace
