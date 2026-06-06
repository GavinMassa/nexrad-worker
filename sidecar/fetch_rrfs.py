"""
fetch_rrfs.py — RRFS upper-air fields via HTTP Range requests.

Replaces fetch_rap.py for all upper-air fields.  Never downloads the full
GRIB2: fetches the small .idx text file, parses byte offsets, then issues
one HTTP Range request per field.  All fields are fetched concurrently.

URL strategy (S3 rrfs_a, valid through ~June 9 2026):
  GRIB2: https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a/rrfs_a.{YYYYMMDD}/{HH}/control/
             rrfs.t{HH}z.prslev.f001.conus.grib2
  IDX:   same + ".idx"

Post-June-9 NOMADS stub (commented out — uncomment when operational):
  GRIB2: https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/
             rrfs.{YYYYMMDD}/{HH}/rrfs.t{HH}z.prslev.3km.f000.conus.grib2
  IDX:   same + ".idx"
"""
import asyncio, gc, logging, tempfile
from pathlib import Path
from datetime import datetime, timedelta

import httpx, cfgrib, numpy as np

log = logging.getLogger(__name__)

TMP_DIR     = Path(tempfile.gettempdir()) / 'sidecar-cache'
TMP_DIR.mkdir(exist_ok=True)
COORDS_CACHE = TMP_DIR / 'rrfs_grid_coords.npz'

RRFS_S3_BASE = 'https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_a'

# RRFS 3km CONUS grid dimensions
RRFS_NY, RRFS_NX = 1059, 1799

# ── Post-June-9 NOMADS URLs (uncomment when operational) ─────────────────────
# RRFS_NOMADS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod'
#
# def _rrfs_urls(dt: datetime) -> tuple[str, str]:
#     ymd = dt.strftime('%Y%m%d'); hh = dt.strftime('%H')
#     base = (f'{RRFS_NOMADS_BASE}/rrfs.{ymd}/{hh}/'
#             f'rrfs.t{hh}z.prslev.3km.f000.conus.grib2')
#     return base, base + '.idx'
# ─────────────────────────────────────────────────────────────────────────────

def _rrfs_urls(dt: datetime) -> tuple[str, str]:
    """Return (grib2_url, idx_url) for the S3 rrfs_a f001 product."""
    ymd = dt.strftime('%Y%m%d')
    hh  = dt.strftime('%H')
    base = (f'{RRFS_S3_BASE}/rrfs_a.{ymd}/{hh}/control/'
            f'rrfs.t{hh}z.prslev.f001.conus.grib2')
    return base, base + '.idx'


# ── Field patterns ────────────────────────────────────────────────────────────
# Each value is a substring matched against wgrib2 idx lines:
#   <rec>:<byte_offset>:<d=YYYYMMDDHH>:<VAR>:<LEVEL>:<TYPE>:
# Patterns include surrounding colons to avoid partial matches.

FIELD_PATTERNS: dict[str, str] = {
    'cape':   ':CAPE:surface:',
    'cin':    ':CIN:surface:',
    'mucape': ':CAPE:180-0 mb above ground:',
    'cape3k': ':CAPE:0-3000 m above ground:',
    'srh1':   ':HLCY:1000-0 m above ground:',
    'ustm':   ':USTM:1000-0 m above ground:',
    'vstm':   ':VSTM:1000-0 m above ground:',
    'u500':   ':UGRD:500 mb:',
    'v500':   ':VGRD:500 mb:',
    'u850':   ':UGRD:850 mb:',
    'v850':   ':VGRD:850 mb:',
    'u925':   ':UGRD:925 mb:',
    'v925':   ':VGRD:925 mb:',
    'u950':   ':UGRD:950 mb:',
    'v950':   ':VGRD:950 mb:',
    't700':   ':TMP:700 mb:',
    't925':   ':TMP:925 mb:',
    'pwat':   ':PWAT:entire atmosphere',  # no trailing colon — suffix varies
}

# If any of these are absent the whole cycle is skipped.
CRITICAL_FIELDS = frozenset({
    'cape', 'cin', 'srh1',
    'u500', 'v500', 'u850', 'v850', 'u925', 'v925',
    't700', 'pwat',
})


# ── IDX parsing ───────────────────────────────────────────────────────────────

def _parse_idx(idx_text: str,
               patterns: dict[str, str]) -> dict[str, tuple[int, int | None]]:
    """
    Parse wgrib2 .idx text and return {key: (start_byte, end_byte)} for
    each pattern.  end_byte is None for the last record (read to EOF).

    Format: <rec>:<byte_offset>:<d=YYYYMMDDHH>:<VAR>:<LEVEL>:<TYPE>:
    Each pattern is matched as a substring of the full idx line.
    The first matching line wins (handles duplicates gracefully).
    """
    # Build sorted list of (byte_offset, raw_line)
    records: list[tuple[int, str]] = []
    for line in idx_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) < 2:
            continue
        try:
            records.append((int(parts[1]), line))
        except ValueError:
            continue
    records.sort(key=lambda x: x[0])

    result: dict[str, tuple[int, int | None]] = {}
    for key, pattern in patterns.items():
        for i, (offset, line) in enumerate(records):
            if pattern in line:
                end = records[i + 1][0] - 1 if i + 1 < len(records) else None
                result[key] = (offset, end)
                break   # first match wins
    return result


# ── Per-field range fetch + cfgrib extraction ─────────────────────────────────

async def _fetch_and_extract(
    key: str,
    start: int,
    end: int | None,
    grib_url: str,
    run_stamp: str,
    client: httpx.AsyncClient,
) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Range-download one GRIB2 record and extract with cfgrib.

    Returns (key, data_array, lats, lons).
    lats/lons are float32 2D arrays on success, None on failure.
    data_array is float32 2D, None on failure.
    """
    range_hdr = f'bytes={start}-{end}' if end is not None else f'bytes={start}-'
    dest = TMP_DIR / f'rrfs_{key}_{run_stamp}.grib2'
    dest.unlink(missing_ok=True)

    try:
        # ── Download the byte range ───────────────────────────────────────────
        async with client.stream('GET', grib_url,
                                 headers={'Range': range_hdr}) as r:
            if r.status_code not in (200, 206):
                log.warning(f'[rrfs] {key}: range request returned {r.status_code}')
                return key, None, None, None
            with open(dest, 'wb') as f:
                async for chunk in r.aiter_bytes(65536):
                    f.write(chunk)

        size = dest.stat().st_size
        if size < 100:
            log.warning(f'[rrfs] {key}: suspiciously small ({size} B)')
            return key, None, None, None

        # ── cfgrib extraction ─────────────────────────────────────────────────
        try:
            ds = cfgrib.open_dataset(str(dest))
            try:
                var_name = list(ds.data_vars)[0]
                arr      = ds[var_name].values
                lats     = ds['latitude'].values.astype(np.float32)
                lons     = ds['longitude'].values.astype(np.float32)

                if arr.ndim == 3:
                    # Log every non-spatial coord so stacking order is auditable.
                    spatial = {'latitude', 'longitude', 'valid_time', 'time', 'step'}
                    for c in ds.coords:
                        if c not in spatial:
                            log.info(f'[rrfs] {key}: stacking coord '
                                     f'{c}={ds.coords[c].values}')
                    log.info(f'[rrfs] {key}: stacked shape={arr.shape} → using [0]')
                    arr = arr[0]

                arr = arr.astype(np.float32)
                log.info(f'[rrfs] {key}: shape={arr.shape} '
                         f'sample={arr.flat[0]:.2f}')
                return key, arr, lats, lons

            finally:
                ds.close()

        except Exception as e:
            log.warning(f'[rrfs] {key}: cfgrib extraction failed: {e}')
            return key, None, None, None

    except Exception as e:
        log.warning(f'[rrfs] {key}: range download failed: {e}')
        return key, None, None, None

    finally:
        dest.unlink(missing_ok=True)
        gc.collect()


# ── Grid coordinate helper ────────────────────────────────────────────────────

def _get_grid_coords(
    live_lats: np.ndarray | None,
    live_lons: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lats_2d, lons_2d) for the RRFS 3km CONUS grid.

    Preference order:
      1. live_lats/live_lons extracted from cfgrib this cycle (most accurate)
      2. Cached .npz from a previous successful cycle
      3. Linspace approximation (last resort — coarser but correct enough for
         bilinear interpolation to the RTMA 2.5km grid)
    """
    if live_lats is not None and live_lons is not None:
        if live_lats.shape == (RRFS_NY, RRFS_NX):
            try:
                np.savez_compressed(str(COORDS_CACHE),
                                    lats=live_lats, lons=live_lons)
                log.info(f'[rrfs] grid coords cached → {COORDS_CACHE}')
            except Exception as e:
                log.warning(f'[rrfs] coord cache write failed: {e}')
            return live_lats, live_lons
        log.warning(f'[rrfs] unexpected coord shape {live_lats.shape}, '
                    f'expected ({RRFS_NY}, {RRFS_NX}) — trying cache')

    if COORDS_CACHE.exists():
        try:
            npz = np.load(str(COORDS_CACHE))
            log.info('[rrfs] grid coords loaded from cache')
            return npz['lats'].astype(np.float32), npz['lons'].astype(np.float32)
        except Exception as e:
            log.warning(f'[rrfs] coord cache load failed: {e}')

    log.warning('[rrfs] falling back to linspace coord approximation')
    lats_1d = np.linspace(21.14, 47.86, RRFS_NY, dtype=np.float32)
    lons_1d = np.linspace(237.28, 299.08, RRFS_NX, dtype=np.float32)
    lats_2d = np.broadcast_to(lats_1d[:, None], (RRFS_NY, RRFS_NX)).copy()
    lons_2d = np.broadcast_to(lons_1d[None, :], (RRFS_NY, RRFS_NX)).copy()
    return lats_2d, lons_2d


# ── Public entry point ────────────────────────────────────────────────────────

async def fetch_rrfs(cycle_dt: datetime) -> dict | None:
    """
    Fetch RRFS upper-air fields for cycle_dt using HTTP Range requests.

    Uses the f001 product so valid_time == cycle_dt.  The run time is
    therefore cycle_dt − 1h; tries up to 4 hourly runs back if the
    most recent is not yet available.

    Returns a dict with keys matching the fetch_rap output convention:
      cape, cin, mucape, cape3k, srh1, ustm, vstm,
      u500, v500, u850, v850, u925, v925, u950, v950,
      t700, t925, pwat, lats_rap, lons_rap

    Non-critical fields (mucape, cape3k, ustm, vstm, u950, v950, t925)
    are set to None rather than aborting the cycle when missing.

    Returns None if no RRFS cycle is available within 4 hours back.
    """
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for offset in range(4):
            run_dt    = cycle_dt - timedelta(hours=offset + 1)
            run_stamp = run_dt.strftime('%Y%m%d_%H')
            valid_hh  = (run_dt + timedelta(hours=1)).strftime('%H')

            grib_url, idx_url = _rrfs_urls(run_dt)
            log.info(f'[rrfs] trying run {run_stamp} '
                     f'(f001 valid at {valid_hh}Z) → {idx_url}')

            # ── Fetch and parse idx ───────────────────────────────────────────
            try:
                r = await client.get(idx_url)
                if r.status_code == 404:
                    log.info(f'[rrfs] {run_stamp}: idx 404 — run not yet posted')
                    continue
                r.raise_for_status()
                idx_text = r.text
            except Exception as e:
                log.warning(f'[rrfs] {run_stamp}: idx fetch failed: {e}')
                continue

            ranges = _parse_idx(idx_text, FIELD_PATTERNS)
            log.info(f'[rrfs] {run_stamp}: idx parsed — '
                     f'{len(ranges)}/{len(FIELD_PATTERNS)} fields located')

            # Bail early if critical fields are missing from the idx itself.
            missing_critical = CRITICAL_FIELDS - set(ranges.keys())
            if missing_critical:
                log.warning(f'[rrfs] {run_stamp}: critical fields absent from idx: '
                             f'{sorted(missing_critical)} — skipping run')
                continue

            # ── Concurrent range fetch + extract ─────────────────────────────
            tasks = [
                _fetch_and_extract(key, start, end, grib_url, run_stamp, client)
                for key, (start, end) in ranges.items()
            ]
            raw_results: list[tuple[str, np.ndarray | None,
                                    np.ndarray | None, np.ndarray | None]] = (
                await asyncio.gather(*tasks)
            )

            # ── Assemble result dict ──────────────────────────────────────────
            data: dict = {}
            live_lats = live_lons = None
            for key, arr, lats, lons in raw_results:
                data[key] = arr
                if live_lats is None and lats is not None:
                    live_lats, live_lons = lats, lons

            # Fields requested but absent from idx → ensure None in dict
            for key in FIELD_PATTERNS:
                data.setdefault(key, None)

            # Check critical fields
            failed_critical = [k for k in CRITICAL_FIELDS if data.get(k) is None]
            if failed_critical:
                log.warning(f'[rrfs] {run_stamp}: critical fields failed extraction: '
                             f'{sorted(failed_critical)} — skipping run')
                continue

            # ── Grid coordinates ──────────────────────────────────────────────
            lats_rap, lons_rap = _get_grid_coords(live_lats, live_lons)
            data['lats_rap'] = lats_rap
            data['lons_rap'] = lons_rap

            n_loaded = sum(1 for k, v in data.items()
                           if k not in ('lats_rap', 'lons_rap') and v is not None)
            log.info(f'[rrfs] {run_stamp}: OK — '
                     f'{n_loaded}/{len(FIELD_PATTERNS)} fields loaded')
            return data

    log.warning('[rrfs] no RRFS cycle available within 4 hours back')
    return None
