"""
GOES-19 satellite pre-processor for iOS.

Fetches the full-resolution GOES-19 ABI CONUS images from NESDIS CDN,
reprojects from ABI fixed-grid projection to Web Mercator (EPSG:3857)
using the GOES-R PUG Vol. 3 §4.2.8.1 inverse mapping, and writes
JPEG files to /app/sidecar-out/ for Node.js to serve.

Products:
  geocolor  — GEOCOLOR (RGB composite): 10848×6136 source
  visible   — Band 02 (0.64μm red visible): 10848×6136 source

Output files:
  /app/sidecar-out/satellite_geocolor.jpg   (~1.5-2.5 MB)
  /app/sidecar-out/satellite_visible.jpg    (~1.0-2.0 MB)
  /app/sidecar-out/satellite_meta.json      — timestamps + bbox

Output resolution: OUT_WIDTH × OUT_HEIGHT pixels, Web Mercator,
clipped to the GOES CONUS bbox used by the iOS SLIDERImageOverlay.

Runtime: ~8-15 seconds per product on 24 vCPU (numpy vectorised,
no scipy, no Metal — pure Python/numpy inverse mapping).
"""

import asyncio
import concurrent.futures
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

OUT_DIR = Path('/app/sidecar-out')
OUT_DIR.mkdir(exist_ok=True)

# ── GOES-19 ABI fixed-grid constants (from GOES-19 ABI L1b RadC NetCDF) ─────
# longitude_of_projection_origin = -75.0° (verified from L1b goes_imager_projection).
# GOES-East nominal slot is 75.0°W (same as GOES-16 before it).
LAMBDA0 = -75.0 * np.pi / 180   # sub-satellite longitude (rad)
H       = 42_164_160.0           # satellite distance from Earth centre (m)
R_EQ    = 6_378_137.0            # equatorial radius (m)
R_POL   = 6_356_752.3141         # polar radius (m)
E2      = 0.00669437999014       # eccentricity²

# CONUS sector ABI fixed-grid scan angle extent (radians).
# Derived from Band 2 L1b: x add_offset=-0.101353, scale=1.4e-5, 10000px (0.5km).
# The 5000×3000 GEOCOLOR is the 1km (2×aggregated) version of the same grid:
#   pixel-0 centre = avg of Band-2 pixels 0 & 1 → -0.101353 + 7e-6 = -0.101346
#   pixel-N centre shifts symmetrically at the other end.
X_MIN =  -0.101346   # west scan angle  (1km CONUS grid, pixel 0 centre)
X_MAX =   0.038626   # east scan angle  (1km CONUS grid, pixel 4999 centre)
Y_MIN =   0.044254   # south elevation angle  (1km CONUS grid, row 2999 centre)
Y_MAX =   0.128226   # north elevation angle  (1km CONUS grid, row 0 centre)

# ── Output grid — Web Mercator, clipped to GOES CONUS satellite bbox ─────────
# These constants MUST match iOS SLIDERImageOverlay / sliderCONUS* constants exactly.
CONUS_LAT_MIN = 14.568
CONUS_LAT_MAX = 53.297
CONUS_LON_MIN = -135.038
CONUS_LON_MAX = -59.975

# Output pixel dimensions in Web Mercator.
# Height is derived from the Mercator y-range so that one pixel represents
# the same Mercator unit in both axes — required for correct tile slicing.
OUT_WIDTH = 5000
_MERC_MAX = np.log(np.tan(np.pi / 4 + CONUS_LAT_MAX * np.pi / 180 / 2))
_MERC_MIN = np.log(np.tan(np.pi / 4 + CONUS_LAT_MIN * np.pi / 180 / 2))
OUT_HEIGHT = int(OUT_WIDTH * (_MERC_MAX - _MERC_MIN) /
                 ((CONUS_LON_MAX - CONUS_LON_MIN) * np.pi / 180))
# ≈ 5000 × 3356 for the CONUS bbox

JPEG_QUALITY = 85

# ── NESDIS CDN source URLs (5000×3000 full-res CONUS) ──────────────────────
NESDIS_URLS = {
    'geocolor': 'https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/GEOCOLOR/5000x3000.jpg',
    'visible':  'https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/02/5000x3000.jpg',
}


def _build_mercator_to_abi_lut() -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute a lookup table mapping every output pixel (OUT_HEIGHT × OUT_WIDTH)
    to normalised source UV coordinates in the ABI fixed-grid image.

    Returns (src_u, src_v): float32 arrays shape (OUT_HEIGHT, OUT_WIDTH).
    Values outside [0,1] indicate pixels outside the GOES disk (set to -1).

    This runs once at module import (~1-2 seconds). The LUT is reused for
    every subsequent reprojection call — zero per-cycle overhead.
    """
    log.info('[satellite] Building Mercator→ABI LUT (%d×%d)…', OUT_HEIGHT, OUT_WIDTH)
    t0 = time.monotonic()

    # Output pixel → geographic lon/lat
    u_out = (np.arange(OUT_WIDTH,  dtype=np.float64) + 0.5) / OUT_WIDTH
    v_out = (np.arange(OUT_HEIGHT, dtype=np.float64) + 0.5) / OUT_HEIGHT

    lon_rad = (CONUS_LON_MIN + u_out * (CONUS_LON_MAX - CONUS_LON_MIN)) * np.pi / 180

    # Web Mercator inverse: v=0 → north (lat_max), v=1 → south (lat_min)
    merc_max = np.log(np.tan(np.pi / 4 + CONUS_LAT_MAX * np.pi / 180 / 2))
    merc_min = np.log(np.tan(np.pi / 4 + CONUS_LAT_MIN * np.pi / 180 / 2))
    merc_y   = merc_max - v_out * (merc_max - merc_min)  # v=0 → north
    lat_rad  = 2 * np.arctan(np.exp(merc_y)) - np.pi / 2

    # Broadcast to 2D grids
    lat2d = lat_rad[:, np.newaxis]   # (H, 1)
    lon2d = lon_rad[np.newaxis, :]   # (1, W)

    # Geodetic → geocentric latitude
    lat_c  = np.arctan((R_POL / R_EQ)**2 * np.tan(lat2d))
    cos_lc = np.cos(lat_c)
    r_c    = R_POL / np.sqrt(1 - E2 * cos_lc**2)

    dl = lon2d - LAMBDA0
    sx =  H - r_c * cos_lc * np.cos(dl)
    sy = -r_c * cos_lc * np.sin(dl)
    sz =  r_c * np.sin(lat_c)

    # Visibility mask (behind-Earth test, GOES-R PUG)
    ratio_sq = (R_EQ / R_POL)**2
    visible  = H * (H - sx) >= sy**2 + ratio_sq * sz**2

    # ABI scan angles
    x_scan = np.arctan(-sy / np.sqrt(sx**2 + sz**2))
    y_scan = np.arctan(sz / sx)

    # Normalise to source image UV [0,1]
    src_u = ((x_scan - X_MIN) / (X_MAX - X_MIN)).astype(np.float32)
    src_v = ((Y_MAX - y_scan) / (Y_MAX - Y_MIN)).astype(np.float32)   # top=north

    # Mask out-of-disk pixels
    in_range = visible & (src_u >= 0) & (src_u <= 1) & (src_v >= 0) & (src_v <= 1)
    src_u = np.where(in_range, src_u, -1.0).astype(np.float32)
    src_v = np.where(in_range, src_v, -1.0).astype(np.float32)

    log.info('[satellite] LUT built in %.1fs', time.monotonic() - t0)
    return src_u, src_v


# Build LUT once at import time.
_LUT_U, _LUT_V = _build_mercator_to_abi_lut()


def _reproject_numpy(src_array: np.ndarray) -> np.ndarray:
    """
    Reproject a source ABI image (H_src × W_src × 3) to the output
    Mercator grid using the pre-computed LUT (_LUT_U, _LUT_V).

    Uses nearest-neighbour sampling — fast and sufficient at this resolution.
    Returns uint8 array (OUT_HEIGHT × OUT_WIDTH × 3).
    """
    h_src, w_src = src_array.shape[:2]

    px = (_LUT_U * (w_src - 1)).astype(np.int32)
    py = (_LUT_V * (h_src - 1)).astype(np.int32)
    mask = (_LUT_U >= 0)

    px = np.clip(px, 0, w_src - 1)
    py = np.clip(py, 0, h_src - 1)

    out = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    out[mask] = src_array[py[mask], px[mask]]
    return out


async def _fetch_image(product: str, client: httpx.AsyncClient) -> bytes | None:
    """Download NESDIS JPEG. Returns raw bytes or None on failure."""
    url = NESDIS_URLS[product]
    try:
        r = await client.get(url, timeout=60.0)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning('[satellite] fetch %s failed: %s', product, e)
        return None


def _process_product(product: str, raw_bytes: bytes) -> bool:
    """
    Decode JPEG, reproject, re-encode at JPEG_QUALITY, write atomically.
    Runs in a ThreadPoolExecutor — does not block the event loop.
    Returns True on success.
    """
    t0 = time.monotonic()
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        src = np.asarray(img, dtype=np.uint8)
        log.info('[satellite] %s source: %dx%d', product, src.shape[1], src.shape[0])

        out     = _reproject_numpy(src)
        out_img = Image.fromarray(out, 'RGB')

        out_path = OUT_DIR / f'satellite_{product}.jpg'
        tmp_path = OUT_DIR / f'satellite_{product}.jpg.tmp'
        out_img.save(str(tmp_path), format='JPEG', quality=JPEG_QUALITY,
                     optimize=True, progressive=True)
        tmp_path.replace(out_path)

        size_kb = out_path.stat().st_size // 1024
        log.info('[satellite] %s → %s (%d KB) in %.1fs',
                 product, out_path.name, size_kb, time.monotonic() - t0)
        return True
    except Exception as e:
        log.error('[satellite] process %s failed: %s', product, e, exc_info=True)
        return False


async def run_satellite_cycle() -> None:
    """
    Fetch and reproject both products concurrently.
    Writes satellite_meta.json atomically after both complete.
    """
    async with httpx.AsyncClient(
        headers={'Cache-Control': 'no-cache'},
        follow_redirects=True,
    ) as client:
        raw_geo, raw_vis = await asyncio.gather(
            _fetch_image('geocolor', client),
            _fetch_image('visible',  client),
        )

    loop    = asyncio.get_running_loop()
    results: dict[str, bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {}
        if raw_geo:
            futs['geocolor'] = loop.run_in_executor(pool, _process_product, 'geocolor', raw_geo)
        if raw_vis:
            futs['visible']  = loop.run_in_executor(pool, _process_product, 'visible',  raw_vis)
        for name, fut in futs.items():
            results[name] = await fut

    meta = {
        'updated':     datetime.now(timezone.utc).isoformat(),
        'geocolor_ok': results.get('geocolor', False),
        'visible_ok':  results.get('visible',  False),
        'lat_min': CONUS_LAT_MIN,
        'lat_max': CONUS_LAT_MAX,
        'lon_min': CONUS_LON_MIN,
        'lon_max': CONUS_LON_MAX,
        'width':   OUT_WIDTH,
        'height':  OUT_HEIGHT,
    }
    tmp = OUT_DIR / 'satellite_meta.json.tmp'
    tmp.write_text(json.dumps(meta))
    tmp.replace(OUT_DIR / 'satellite_meta.json')
    log.info('[satellite] cycle done: geocolor=%s visible=%s',
             results.get('geocolor'), results.get('visible'))


async def satellite_worker() -> None:
    """
    Background coroutine: runs every 5 minutes, never crashes the process.
    First run is immediate (on startup).
    """
    INTERVAL = 5 * 60
    while True:
        try:
            await run_satellite_cycle()
        except Exception as e:
            log.error('[satellite] worker error: %s', e, exc_info=True)
        await asyncio.sleep(INTERVAL)
