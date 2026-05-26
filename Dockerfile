# Railway build — explicit Dockerfile to guarantee wgrib2 is on the runtime PATH.
#
# Multi-stage: compile wgrib2 from NOAA source in a builder stage,
# then copy the static binary into the slim runtime image.

# ── Stage 1: build wgrib2 from source ────────────────────────────────────────
FROM debian:bookworm-slim AS wgrib2-builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        wget \
        ca-certificates \
        libaec-dev \
 && rm -rf /var/lib/apt/lists/*

# Download and compile wgrib2 from NOAA source.
WORKDIR /build
RUN wget -q https://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz \
 && tar xzf wgrib2.tgz \
 && cd grib2 \
 && export CC=gcc FC=gfortran \
 && make USE_AEC=0 \
 && cp wgrib2/wgrib2 /usr/local/bin/wgrib2 \
 && strip /usr/local/bin/wgrib2

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM node:20-bookworm-slim

# Copy the compiled wgrib2 binary from the builder stage.
COPY --from=wgrib2-builder /usr/local/bin/wgrib2 /usr/local/bin/wgrib2

# System packages:
#   ca-certificates — for HTTPS fetches to NOMADS
#   curl            — useful for debugging from `railway run`
#   libaec0         — runtime dependency for wgrib2 (adaptive entropy coding)
#   libgomp1        — OpenMP runtime (wgrib2 uses it for parallel decoding)
#   libgfortran5    — Fortran runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libaec0 \
        libgomp1 \
        libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server.js"]

