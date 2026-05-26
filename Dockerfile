# Railway build — explicit Dockerfile to guarantee wgrib2 is on the runtime PATH.
#
# Base image: official Node 20 on Debian bookworm-slim.
# wgrib2 (from the Debian 12 main repo) is installed into /usr/bin.

FROM node:20-bookworm-slim

# System packages:
#   wgrib2          — single-pass GRIB2 field extraction replacing eccodes workers
#   ca-certificates — for HTTPS fetches to NOMADS
#   curl            — useful for debugging from `railway run`
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        wgrib2 \
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
