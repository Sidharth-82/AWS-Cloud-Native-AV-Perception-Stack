FROM python:3.11-slim

# Offline processing image (Phase 1 steps 8-10). Deliberately NOT the capture
# image: no carla wheel, no CUDA, no GPU runtime. This half is pure numpy/PIL
# CPU work, so pinning it to python 3.7 (which the carla wheel forces on the
# capture side) would be a needless constraint.
#
# 3.11 specifically, not 3.7: offline/parser.py uses `tuple[dict, ...]` builtin
# generics (3.9+), and `parser` was a STDLIB module name until 3.10 -- on 3.7
# `import parser` can resolve to the built-in extension module instead of ours.

RUN pip install --no-cache-dir \
      numpy==1.26.4 \
      Pillow==10.3.0 \
      boto3==1.34.84

# Unbuffered stdout/stderr so print()/tracebacks appear live and survive a hard
# crash (OOM). Docker's non-TTY pipe block-buffers Python output otherwise.
ENV PYTHONUNBUFFERED=1

# common/ holds config_loader + utils, shared with the capture image. Put it on
# the path rather than duplicating it, so both halves read configs through one
# loader and cannot drift on how documentation keys are stripped.
ENV PYTHONPATH=/workspace/src/common

WORKDIR /workspace
