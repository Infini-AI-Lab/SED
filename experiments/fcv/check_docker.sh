#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found in PATH"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon not running. Start Docker Desktop / Colima first."
  exit 1
fi

# Pull smoke image if missing (amd64 on Apple Silicon)
IMAGE="${FCV_DOCKER_TEST_IMAGE:-swebench/sweb.eval.x86_64.django_1776_django-10914:latest}"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Pulling ${IMAGE} ..."
  docker pull --platform linux/amd64 "${IMAGE}"
fi

docker run --rm --platform linux/amd64 "${IMAGE}" echo docker_ok >/dev/null
echo "Docker OK (${IMAGE})"
