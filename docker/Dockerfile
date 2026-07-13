# Sandbox image for `executor: docker` tasks.
# Build once:  docker build -t ramanujan-sandbox docker/
FROM python:3.12-slim

RUN pip install --no-cache-dir scikit-learn pandas numpy

# Generated scripts run with --network=none, so everything they need
# must already be in this image.
