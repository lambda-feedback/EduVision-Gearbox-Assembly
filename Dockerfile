FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12 AS builder

RUN python -m pip install --no-cache-dir poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

# Install deps (torch/torchvision will be resolved from pytorch-cpu source via pyproject.toml)
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root

# Sanity check: ensure torch is CPU-only (cuda should be None)
RUN python -c "import torch; print('torch', torch.__version__); print('torch.cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available())"


FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Precompile python files for faster startup
RUN python -m compileall -q .

# Copy the evaluation function to the app directory
COPY evaluation_function ./evaluation_function

# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

# Interface / logging
ENV FUNCTION_INTERFACE="file"
ENV LOG_LEVEL="debug"
