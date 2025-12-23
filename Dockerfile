# syntax=docker/dockerfile:1

ARG PIXI_VERSION=0.62.0
ARG PIXI_ENV=default

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build
ARG PIXI_ENV
WORKDIR /app

# Only copy manifests first for better layer caching
COPY pyproject.toml pixi.lock /app/

# Install the selected environment from the lockfile
RUN pixi install --locked -e "${PIXI_ENV}"

# Generate a shell-hook that matches the environment we are shipping
RUN pixi shell-hook -e "${PIXI_ENV}" -s bash > /shell-hook

# Copy the repo last
COPY . /app

FROM ubuntu:24.04 AS runtime
ARG PIXI_ENV
ENV PIXI_ENV="${PIXI_ENV}"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates tini \
  && rm -rf /var/lib/apt/lists/*

# Copy exactly the env we built
COPY --from=build /app/.pixi/envs/${PIXI_ENV} /app/.pixi/envs/${PIXI_ENV}
COPY --from=build /shell-hook /shell-hook

# Repo contents (scripts, weights, etc.)
COPY . /app

# Entrypoint activates env then runs your command / shell
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/usr/bin/tini","--","/entrypoint.sh"]
CMD ["bash"]
