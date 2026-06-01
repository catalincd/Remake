#!/usr/bin/env bash
# Launch the ROCm PyTorch container for this project.
#
# Mounts the *parent* of this repo (…/Workspace) at the same absolute path
# inside the container, so the dataset's fragment symlinks (which point at the
# sibling FFT repo) resolve correctly, and drops you into the repo with the
# RX 6750 XT (gfx1031) gfx-override already set.
#
# Usage:
#   ./run_docker_torch.sh                  # interactive shell in the repo
#   ./run_docker_torch.sh ./run.sh setup   # run a command, then exit
#   ./run_docker_torch.sh ./run.sh zoo-smoke
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOUNT_DIR="$(dirname "$REPO_DIR")"      # …/Workspace  (covers Remake + FFT data)
IMAGE="${ROCM_IMAGE:-rocm/pytorch:latest}"

docker run -it --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$MOUNT_DIR":"$MOUNT_DIR" \
    -w "$REPO_DIR" \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    "${@:-/bin/bash}"
