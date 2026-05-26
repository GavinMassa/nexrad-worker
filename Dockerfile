# Railway build — wgrib2 compiled from source.
#
# wgrib2 is a NOAA/NCEP tool with no official Debian package ("E: Unable to
# locate package wgrib2").  We compile it in a disposable builder stage using
# the self-contained NCEP source tarball, then copy only the resulting binary
# into the production image.  Build tools and intermediate objects (hundreds of
# MB) never land in the final layer.
#
# Build time estimate: ~3-5 min (Makefile fetches jasper/libpng inline via wget).
# Parallelism: -j$(nproc) uses all available builder cores.

# ── Stage 1: compile wgrib2 from NCEP source ────────────────────────────────
FROM node:20-bookworm-slim AS wgrib2-builder

# Compiler chain + wget for the Makefile's inline library downloads.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gcc gfortran g++ make wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The NCEP wgrib2.tgz is a self-contained bundle: its Makefile downloads and
# statically links jasper/libpng, so no extra apt packages are needed here.
#
# Optional components disabled to cut build time:
#   USE_NETCDF4/3 = 0  — skip HDF5 + NetCDF (adds ~10 min; not used by rap.js)
#   USE_IPOLATES  = 0  — skip NCEP spatial interpolation library
#   USE_OPENMP    = 0  — single-threaded is fine for our sequential extractions
#   USE_AEC       = 0  — adaptive entropy coding not present in RAP GRIB2 files
#
# GNU Make command-line assignments override Makefile-internal definitions,
# so these flags work regardless of how each variable is set in the Makefile.
RUN wget -q \
      'https://www.ftp.ncep.noaa.gov/pub/wd51we/wgrib2/wgrib2.tgz' \
      -O /tmp/wgrib2.tgz \
 && tar xzf /tmp/wgrib2.tgz -C /tmp \
 && cd /tmp/grib2 \
 && make -j"$(nproc)" \
         CC=gcc FC=gfortran \
         USE_NETCDF4=0 \
         USE_NETCDF3=0 \
         USE_IPOLATES=0 \
         USE_OPENMP=0 \
         USE_AEC=0 \
 && cp /tmp/grib2/wgrib2/wgrib2 /usr/local/bin/wgrib2 \
 && rm -rf /tmp/grib2 /tmp/wgrib2.tgz

# ── Stage 2: production runtime ─────────────────────────────────────────────
FROM node:20-bookworm-slim

# Runtime libraries the wgrib2 binary links against at run time.
# The NCEP Makefile statically links jasper/libpng, so only the gfortran
# stdlib and standard C libs are required here.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgfortran5 \
        libquadmath0 \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

# Only the compiled binary comes from the builder — no build tools in prod.
COPY --from=wgrib2-builder /usr/local/bin/wgrib2 /usr/local/bin/wgrib2

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY . .

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server.js"]
