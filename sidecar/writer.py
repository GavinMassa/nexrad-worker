import json, logging
import numpy as np
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)
OUT_DIR = Path('/tmp/sidecar-out')
OUT_DIR.mkdir(exist_ok=True)

def write_output(grids: dict, cycle_dt: datetime) -> None:
    """
    Write completed grids to OUT_DIR atomically (write → .tmp, then os.replace).
    Node.js reads these files and serves them via /rap/blend/all.

    grids: dict containing:
        - param arrays: str → np.ndarray (float32, shape ny×nx)
        - 'lats': np.ndarray (float32, shape ny×nx)
        - 'lons': np.ndarray (float32, shape ny×nx)
    cycle_dt: the valid time for this analysis cycle.
    """
    params = [k for k in grids if k not in ('lats', 'lons')]
    ny, nx = grids['lats'].shape

    # Write each parameter grid atomically: .tmp → final.
    for param in params:
        arr = grids[param].astype(np.float32)
        final = OUT_DIR / f'{param}.bin'
        tmp   = OUT_DIR / f'{param}.bin.tmp'
        arr.tofile(str(tmp))
        tmp.replace(final)  # atomic on POSIX; Node never reads a partial file

    # Write meta.json atomically last, after all .bin files are in place.
    # Node should check meta.json to know the grid is complete.
    meta = {
        'nx':         nx,
        'ny':         ny,
        'lat_min':    float(grids['lats'].min()),
        'lat_max':    float(grids['lats'].max()),
        'lon_min':    float(grids['lons'].min()),
        'lon_max':    float(grids['lons'].max()),
        'valid_time': cycle_dt.isoformat(),
        'params':     params,
        'source':     'RTMA-only (step 1 — blend not yet implemented)',
    }
    meta_final = OUT_DIR / 'meta.json'
    meta_tmp   = OUT_DIR / 'meta.json.tmp'
    meta_tmp.write_text(json.dumps(meta))
    meta_tmp.replace(meta_final)  # atomic — Node sees either old or new, never partial

    log.info(f'Wrote {len(params)} grids to {OUT_DIR}: {params}')
