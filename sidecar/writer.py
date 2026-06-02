import json, logging, shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom

log = logging.getLogger(__name__)
OUT_DIR       = Path('/app/sidecar-out')
ARCHIVE_HOURS = 6          # retain the last N hourly blend cycles
OUT_DIR.mkdir(exist_ok=True)

# Downsample factor applied to all param grids before writing.
# RTMA native is 1597×2345 (~15 MB per Float32 grid × 8 params = 120 MB body).
# 0.25 → roughly 400×586 (~3.7 MB total), closer to RAP-native resolution
# and matches what the iOS parser already handles for /rap/all.
# Bilinear (order=1) interpolation — visually smooth, no ringing.
DOWNSAMPLE_FACTOR = 0.5

# GOES-19 CONUS sector bbox — must exactly match iOS SLIDERImageOverlay constants:
#   sliderCONUSLatMin = 14.568
#   sliderCONUSLatMax = 53.297
#   sliderCONUSLonMin = -135.038
#   sliderCONUSLonMax = -59.975
# All blend grids are clipped to this bbox before writing so the overlay
# co-registers perfectly with the satellite background on iOS.
CLIP_LAT_MIN: float = 14.568
CLIP_LAT_MAX: float = 53.297
CLIP_LON_MIN: float = -135.038
CLIP_LON_MAX: float = -59.975


def write_output(grids: dict, cycle_dt: datetime) -> None:
    """
    Write completed grids to OUT_DIR atomically (write → .tmp, then os.replace).
    Node.js reads these files and serves them via /rap/blend/all.

    grids: dict containing:
        - param arrays: str → np.ndarray (float32, shape ny×nx)
        - 'lats': np.ndarray (float32, shape ny×nx)
        - 'lons': np.ndarray (float32, shape ny×nx)
    cycle_dt: the valid time for this analysis cycle.

    Pipeline per call:
      1. Normalise lons 0..360 → -180..180
      2. Determine S→N flip flag (before clip — clip doesn't change scan order)
      3. Clip all grids to GOES satellite bbox (CLIP_LAT/LON_MIN/MAX)
      4. Flip clipped grids N→S if needed
      5. Downsample by DOWNSAMPLE_FACTOR (bilinear)
      6. Write .bin files atomically, then meta.json
    """
    params = [k for k in grids if k not in ('lats', 'lons')]
    src_lats = grids['lats']
    src_lons = grids['lons']
    src_ny, src_nx = src_lats.shape

    # ── Coordinate normalisation ─────────────────────────────────────────────
    # RTMA stores longitudes in 0..360 (eastward from Greenwich). iOS/MapKit
    # expects -180..180. Without this fix, CLLocationCoordinate2D(longitude:221)
    # falls outside Mercator → MKMapPoint(-1,-1) → zero-size overlay rect
    # → nothing renders.
    lons_180 = np.where(src_lons > 180.0, src_lons - 360.0, src_lons)

    # ── Flip flag (determined on full grid, before clip) ─────────────────────
    # RTMA's native scan order has row 0 = SW corner (lat_min, southernmost).
    # iOS RAPGridOverlay maps texture row 0 → lat_max (northernmost), so we
    # must flip the data N→S before writing.
    # The clip step below does not change scan order — flip_rows remains valid.
    flip_rows = bool(src_lats[0, 0] < src_lats[-1, 0])
    if flip_rows:
        log.info('RTMA grid is S→N (row 0 = lat_min); flipping rows for iOS')

    # ── Spatial clip to GOES satellite bbox ──────────────────────────────────
    # Use centre column/row for 1D lat/lon extraction — RTMA is curvilinear
    # (Lambert Conformal), so the outer columns curve and are not representative.
    row_lats = src_lats[:, src_lats.shape[1] // 2]   # centre-column lat per row
    col_lons = lons_180[lons_180.shape[0] // 2, :]   # centre-row lon per column

    row_mask = (row_lats >= CLIP_LAT_MIN) & (row_lats <= CLIP_LAT_MAX)
    col_mask = (col_lons >= CLIP_LON_MIN) & (col_lons <= CLIP_LON_MAX)

    src_lats_clipped = src_lats[np.ix_(row_mask, col_mask)]
    lons_180_clipped = lons_180[np.ix_(row_mask, col_mask)]
    src_ny_clip, src_nx_clip = src_lats_clipped.shape

    log.info(f'Clip: {src_ny}×{src_nx} → {src_ny_clip}×{src_nx_clip} '
             f'(lat {src_lats_clipped.min():.2f}–{src_lats_clipped.max():.2f}, '
             f'lon {lons_180_clipped.min():.2f}–{lons_180_clipped.max():.2f})')

    # ── Param loop: clip → flip → downsample → write ─────────────────────────
    out_ny, out_nx = None, None

    for param in params:
        src = grids[param].astype(np.float32)
        # Clip to satellite bbox (row + column boolean index)
        src = src[np.ix_(row_mask, col_mask)]
        # Flip S→N if needed (same flag — clipping doesn't change scan order)
        if flip_rows:
            src = np.flipud(src)
        out = zoom(src, DOWNSAMPLE_FACTOR, order=1).astype(np.float32)
        if out_ny is None:
            out_ny, out_nx = out.shape
            log.info(f'Downsample: {src_ny_clip}×{src_nx_clip} → {out_ny}×{out_nx} '
                     f'(factor={DOWNSAMPLE_FACTOR})')

        final = OUT_DIR / f'{param}.bin'
        tmp   = OUT_DIR / f'{param}.bin.tmp'
        out.tofile(str(tmp))
        tmp.replace(final)  # atomic on POSIX; Node never reads a partial file

    # Fallback if params was empty (defensive — shouldn't happen).
    if out_ny is None:
        out_ny, out_nx = src_ny_clip, src_nx_clip

    # ── meta.json ─────────────────────────────────────────────────────────────
    # Bbox from clipped (not full-domain, not downsampled) lats/lons.
    # These match the GOES satellite bbox so iOS overlay co-registers exactly.
    lat_min = float(src_lats_clipped.min())
    lat_max = float(src_lats_clipped.max())
    lon_min = float(lons_180_clipped.min())
    lon_max = float(lons_180_clipped.max())
    nx_new  = int(out_nx)
    ny_new  = int(out_ny)

    meta = {
        'nx':         nx_new,
        'ny':         ny_new,
        'lat_min':    lat_min,
        'lat_max':    lat_max,
        'lon_min':    lon_min,
        'lon_max':    lon_max,
        'valid_time': cycle_dt.isoformat(),
        'params':     params,
        'source':     f'RTMA+RAP blend, clipped+downsampled {DOWNSAMPLE_FACTOR}× '
                      f'from {src_ny}×{src_nx} → {src_ny_clip}×{src_nx_clip}',
    }
    meta_final = OUT_DIR / 'meta.json'
    meta_tmp   = OUT_DIR / 'meta.json.tmp'
    meta_tmp.write_text(json.dumps(meta))
    meta_tmp.replace(meta_final)  # atomic — Node sees either old or new, never partial

    log.info(f'meta bbox: lat {lat_min:.2f}–{lat_max:.2f}, lon {lon_min:.2f}–{lon_max:.2f}, '
             f'nx={nx_new}, ny={ny_new}')
    log.info(f'Wrote {len(params)} grids ({ny_new}×{nx_new}) to {OUT_DIR}: {params}')

    # ── Archive this cycle ────────────────────────────────────────────────────
    # Copies the freshly-written flat files into OUT_DIR/YYYYMMDDHH/ so the
    # Node.js server can serve historical cycles at GET /rap/blend/YYYYMMDDHH.
    # The flat files in OUT_DIR remain untouched — existing /rap/blend/all
    # endpoint continues to serve the latest cycle with zero disruption.
    hour_key  = cycle_dt.strftime('%Y%m%d%H')
    cycle_dir = OUT_DIR / hour_key
    cycle_dir.mkdir(exist_ok=True)

    for param in params:
        src = OUT_DIR / f'{param}.bin'
        dst = cycle_dir / f'{param}.bin'
        tmp = cycle_dir / f'{param}.bin.tmp'
        shutil.copy2(str(src), str(tmp))
        tmp.replace(dst)

    cycle_meta_tmp   = cycle_dir / 'meta.json.tmp'
    cycle_meta_final = cycle_dir / 'meta.json'
    cycle_meta_tmp.write_text(json.dumps(meta))
    cycle_meta_tmp.replace(cycle_meta_final)

    log.info(f'Archived cycle {hour_key} → {cycle_dir} ({len(params)} params × {ny_new}×{nx_new})')

    # ── Prune cycles beyond ARCHIVE_HOURS ────────────────────────────────────
    # Dirs are named YYYYMMDDHH (10 digits); lexicographic sort = time order.
    existing_dirs = sorted(
        [d for d in OUT_DIR.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name.isdigit()],
        key=lambda d: d.name,
    )
    while len(existing_dirs) > ARCHIVE_HOURS:
        oldest = existing_dirs.pop(0)
        shutil.rmtree(str(oldest), ignore_errors=True)
        log.info(f'Pruned old archive dir: {oldest.name}')

    # ── history.json ──────────────────────────────────────────────────────────
    # Written atomically so Node.js can return available hours without listing
    # the filesystem on every request.
    available = sorted(
        d.name for d in OUT_DIR.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name.isdigit()
    )
    history = {
        'hours':      available,
        'current':    hour_key,
        'updated_at': cycle_dt.isoformat(),
    }
    hist_tmp   = OUT_DIR / 'history.json.tmp'
    hist_final = OUT_DIR / 'history.json'
    hist_tmp.write_text(json.dumps(history))
    hist_tmp.replace(hist_final)
    log.info(f'history.json: {len(available)} hours available — {available}')
