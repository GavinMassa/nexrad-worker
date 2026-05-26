import json, logging
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom

log = logging.getLogger(__name__)
OUT_DIR = Path('/tmp/sidecar-out')
OUT_DIR.mkdir(exist_ok=True)

# Downsample factor applied to all param grids before writing.
# RTMA native is 1597×2345 (~15 MB per Float32 grid × 8 params = 120 MB body).
# 0.25 → roughly 400×586 (~3.7 MB total), closer to RAP-native resolution
# and matches what the iOS parser already handles for /rap/all.
# Bilinear (order=1) interpolation — visually smooth, no ringing.
DOWNSAMPLE_FACTOR = 0.25

def write_output(grids: dict, cycle_dt: datetime) -> None:
    """
    Write completed grids to OUT_DIR atomically (write → .tmp, then os.replace).
    Node.js reads these files and serves them via /rap/blend/all.

    grids: dict containing:
        - param arrays: str → np.ndarray (float32, shape ny×nx)
        - 'lats': np.ndarray (float32, shape ny×nx)
        - 'lons': np.ndarray (float32, shape ny×nx)
    cycle_dt: the valid time for this analysis cycle.

    Each param grid is downsampled by DOWNSAMPLE_FACTOR (bilinear) before
    being written. lats/lons themselves are not written — only their min/max
    go into meta.json's bbox, and those corner values are preserved across
    downsampling.
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

    # RTMA's native scan order has row 0 = SW corner (lat_min, southernmost).
    # iOS RAPGridOverlay maps texture row 0 → lat_max (northernmost), so we
    # must flip the data N→S before writing. Same bug we hit for /rap/all,
    # fixed there with wgrib2's -order we:ns flag.
    flip_rows = bool(src_lats[0, 0] < src_lats[-1, 0])
    if flip_rows:
        log.info('RTMA grid is S→N (row 0 = lat_min); flipping rows for iOS')

    # Downsample one param first to learn the resulting shape so meta.json
    # carries accurate dimensions (zoom rounds; 1597 * 0.25 = 399.25 → 399).
    out_ny, out_nx = None, None

    for param in params:
        src = grids[param].astype(np.float32)
        if flip_rows:
            src = np.flipud(src)
        out = zoom(src, DOWNSAMPLE_FACTOR, order=1).astype(np.float32)
        if out_ny is None:
            out_ny, out_nx = out.shape
            log.info(f'Downsample: {src_ny}×{src_nx} → {out_ny}×{out_nx} '
                     f'(factor={DOWNSAMPLE_FACTOR})')

        final = OUT_DIR / f'{param}.bin'
        tmp   = OUT_DIR / f'{param}.bin.tmp'
        out.tofile(str(tmp))
        tmp.replace(final)  # atomic on POSIX; Node never reads a partial file

    # Fallback if params was empty (defensive — shouldn't happen).
    if out_ny is None:
        out_ny, out_nx = src_ny, src_nx

    # Write meta.json atomically last, after all .bin files are in place.
    # Node checks meta.json as a readiness signal — if it's present, every
    # .bin is fully written.
    meta = {
        'nx':         int(out_nx),
        'ny':         int(out_ny),
        'lat_min':    float(src_lats.min()),
        'lat_max':    float(src_lats.max()),
        'lon_min':    float(lons_180.min()),
        'lon_max':    float(lons_180.max()),
        'valid_time': cycle_dt.isoformat(),
        'params':     params,
        'source':     f'RTMA+RAP blend, downsampled {DOWNSAMPLE_FACTOR}× from {src_ny}×{src_nx}',
    }
    meta_final = OUT_DIR / 'meta.json'
    meta_tmp   = OUT_DIR / 'meta.json.tmp'
    meta_tmp.write_text(json.dumps(meta))
    meta_tmp.replace(meta_final)  # atomic — Node sees either old or new, never partial

    log.info(f'Wrote {len(params)} grids ({out_ny}×{out_nx}) to {OUT_DIR}: {params}')
