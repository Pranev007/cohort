# syntax=docker/dockerfile:1.7
# Multi-stage: build wheels once, ship a slim non-root runtime.

FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

# Apply outstanding Debian security patches. The python:*-slim tags are rebuilt
# on their own cadence, so a fresh pull routinely lags the security archive by
# days -- long enough for Trivy to (correctly) fail the build on CVEs that
# already have fixes available.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install into a virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
 && pip install '.[api]'


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="cohort" \
      org.opencontainers.image.description="Peer-baseline anomaly detection for unstructured enterprise data" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

# Same security patches in the runtime stage -- the builder's are discarded with
# the build layer, and this is the image that actually ships.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Non-root. The image writes only to /app/artifacts, which is a volume.
RUN useradd --create-home --uid 10001 cohort
COPY --from=builder /opt/venv /opt/venv

# Patch the packages the image scan flags, in *both* interpreters.
#
# This image has two independent site-packages trees: the base image's system
# one at /usr/local/lib, and the venv copied from the builder. ENV PATH points at
# the venv from the top of this stage, so a bare `pip install --upgrade` only
# ever reaches one of them -- while Trivy scans the whole filesystem and finds
# whichever was missed. Both are addressed explicitly, by absolute path.
RUN /usr/local/bin/python -m pip install --no-cache-dir --upgrade \
        "setuptools>=78.1.1" \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade \
        "setuptools>=78.1.1" "msgpack>=1.2.1" \
 && /opt/venv/bin/pip check

WORKDIR /app
COPY --chown=cohort:cohort configs ./configs
RUN mkdir -p /app/artifacts && chown -R cohort:cohort /app
USER cohort

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

# Default to the API. Override for batch work, e.g.
#   docker run --rm -v $PWD/artifacts:/app/artifacts cohort:latest cohort demo
CMD ["uvicorn", "cohort.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
