FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

# Remove the broken Yarn repository and its keys before apt-get runs
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && rm -f /etc/apt/trusted.gpg.d/yarn.gpg

# OpenMP runtime used by numpy/scipy/sklearn/torch CPU
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

USER vscode
RUN mkdir -p /home/vscode/.cache/uv
