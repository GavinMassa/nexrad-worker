# Railway build — explicit Dockerfile because Nixpacks `aptPkgs` didn't get
# `libeccodes-tools` (and therefore `grib_get_data`) onto the runtime PATH.
#
# Base image: official Node 20 on Debian bookworm-slim. `libeccodes-tools`
# installs `grib_get_data` into /usr/bin which is on PATH for every layer.

FROM node:20-bookworm-slim

# System packages:
#   - libeccodes-tools : grib_get_data CLI used by rap-worker.js
#   - python3          : reserved for future Python parsers
#   - ca-certificates  : for https fetches to NOMADS / IEM
#   - curl             : occasionally useful for debugging from `railway run`
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libeccodes-tools \
        python3 \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server.js"]
