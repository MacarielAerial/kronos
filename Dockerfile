#
# Multi Stage: Dev Image
# Multi Stage: Dev Image
#
FROM python:3.10 AS dev
FROM python:3.10 AS dev

# Set environemntal variables
ENV PATH = "${PATH}:/home/poetry/bin"
ENV POETRY_VIRTUALENVS_IN_PROJECT=1

# Install graphviz, git and git lfs
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    git \
    git-lfs

# Install poetry
RUN mkdir -p /home/poetry && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/poetry python3 - && \
    poetry self add poetry-plugin-up

#
# Multi Stage: Bake Image
#

FROM dev AS bake
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/poetry python3 - && \
    poetry self add poetry-plugin-up

#
# Multi Stage: Bake Image
#

FROM dev AS bake

# Make working directory
RUN mkdir -p /app

# TODO: Only copy necessary files
# TODO: Only copy necessary files
COPY . /app

# Set working directory
WORKDIR /app

# Install python dependencies in container
RUN poetry install --without dev,vis
RUN poetry install --without dev,vis

#
# Multi Stage: Runtime Image
#


FROM python:3.10-slim AS runtime

# Copy over baked environment
COPY --from=bake /app /app
COPY --from=bake /app /app

# Set 
WORKDIR /app

# Set executables in PATH
ENV PATH="/app/.venv/bin:$PATH"

# TODO: Add a command to start a FastAPI service
# ENTRYPOINT
