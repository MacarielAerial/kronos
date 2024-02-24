#
# Multi Stage: Base Image
#
FROM python:3.10 AS base

# Set environemntal variables
ENV PATH = "${PATH}:/home/poetry/bin"
ENV POETRY_VIRTUALENVS_IN_PROJECT=1

# Set argument for environment with default value
ARG ENVIRONMENT=live

# Install graphviz and git
RUN apt update && apt install -y \
    graphviz-dev

# Install git lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs

# Install poetry
RUN mkdir -p /home/poetry && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/poetry python3 -

# Make working directory
RUN mkdir -p /app

# TODO: Only copy necessary files
COPY . /app

# Set working directory
WORKDIR /app

# Install python dependencies in container
RUN if [ "$ENVIRONMENT" = "dev" ]; then \
        poetry install; \
    else \
        poetry install --without dev,vis; \
    fi

#
# Multi Stage: Runtime Image
#
FROM python:3.10-slim AS runtime

# Copy over baked environment
COPY --from=base /app /app

# Set 
WORKDIR /app

# Set executables in PATH
ENV PATH="/app/.venv/bin:$PATH"

# TODO: Add a command to start a FastAPI service
# ENTRYPOINT
