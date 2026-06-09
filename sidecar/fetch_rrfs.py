"""
fetch_rrfs.py — RRFS upper-air fields via HTTP Range requests.

Never downloads a full GRIB2: fetches the small .idx text files, parses
byte offsets, then issues one HTTP Range request per field.  All fields
are fetched concurrently.

URL strategy (rrfs_public bucket, f000 products):
  prslev: https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public/
              rrfs.{YYYYMMDD}/{HH}/rrfs.t{HH}z.prslev.3km.f000.conus.grib2
  twodfd: same prefix / rrfs.t{HH}z.2dfld.3km.f000.conus.grib2
"""
import asyncio, gc, logging, tempfile
from pathlib import Path
from datetime import datetime, timedelta

import httpx, cfgrib, numpy as np

log = logging.getLogger(__name__)

TMP_DIR      = Path(tempfile.gettempdir()) / 'sidecar-cache'
TMP_DIR.mkdir(exist_ok=True)
COORDS_CACHE = TMP_DIR / 'rrfs_grid_coords.npz'

RRFS_PUBLIC_BASE = 'https://noaa-rrfs-pds.s3.amazonaws.com/rrfs_public'

# RRFS 3km CONUS grid dimensions
RRFS_NY, RRFS_NX = 1059, 1799


# ── URL builder ───────────────────────────────────────────────────────────────

def _source_urls(ymd: str, hh: str) -> tuple[str, str, str, str]:
    """Return (prslev_grib, prslev_idx, twodfd_grib, twodfd_idx)."""
    base    = f'{RRFS_PUBLIC_BASE}/rrfs.{ymd}/{hh}/'
    prslev  = base + f'rrfs.t{hh}z.prslev.3km.f000.conus.grib2'
    twodfd  = base + f'rrfs.t{hh}z.2dfld.3km.f000.conus.grib2'
    return prslev, prslev + '.idx', twodfd, twodfd + '.idx'


# ── Field patterns ────────────────────────────────────────────────────────────
# Each value is a substring matched against wgrib2 idx lines.
# Patterns include surrounding colons to avoid partial matches.

# Pressure-level fields — prslev file
FIELDS_PRSLEV: list[tuple[str, str]] = [
    ('u500', ':UGRD:500 mb:'),
    ('v500', ':VGRD:500 mb:'),
    ('u850', ':UGRD:850 mb:'),
    ('v850', ':VGRD:850 mb:'),
    ('u925', ':UGRD:925 mb:'),
    ('v925', ':VGRD:925 mb:'),
    ('u950', ':UGRD:950 mb:'),
    ('v950', ':VGRD:950 mb:'),
    ('t700', ':TMP:700 mb:'),
    ('t925', ':TMP:925 mb:'),
    ('t850',  ':TMP:850 mb:'),
    ('t500',  ':TMP:500 mb:'),
    ('rh500', ':RH:500 mb:'),
    ('rh700', ':RH:700 mb:'),
    ('rh850', ':RH:850 mb:'),
    ('rh925', ':RH:925 mb:'),
    ('rh950', ':RH:950 mb:'),
    ('t950',  ':TMP:950 mb:'),
    # Upper-troposphere levels — needed to reach the EL (typically 200–250 mb).
    # Without these, CAPE integration truncates at 500 mb and misses 1500–3000 J/kg
    # of positive buoyancy in the 500→200 mb layer where Tv_parcel >> Tv_env.
    ('t600',  ':TMP:600 mb:'),
    ('rh600', ':RH:600 mb:'),
    ('t400',  ':TMP:400 mb:'),
    ('rh400', ':RH:400 mb:'),
    ('t300',  ':TMP:300 mb:'),
    ('rh300', ':RH:300 mb:'),
    ('t200',  ':TMP:200 mb:'),
    ('rh200', ':RH:200 mb:'),
]

# Surface / derived fields — 2dfld file
FIELDS_2DFLD: list[tuple[str, str]] = [
    ('cape',   ':CAPE:surface:'),
    ('cin',    ':CIN:surface:'),
    ('mucape', ':CAPE:180-0 mb above ground:'),
    ('cape3k', ':CAPE:0-3000 m above ground:'),
    ('srh1',   ':HLCY:1000-0 m above ground:'),
    ('ustm',   ':USTM:6000-0 m above ground:'),
    ('vstm',   ':VSTM:6000-0 m above ground:'),
    ('pwat',   ':PWAT:entire atmosphere (considered as a single layer):'),
]

# All fields combined (for result initialisation)
ALL_FIELDS: list[tuple[str, str]] = FIELDS_PRSLEV + FIELDS_2DFLD

# If any of these are absent the whole cycle is skipped.
CRITICAL_FIELDS = frozenset({
    'cape', 'cin', 'srh1',
    'u500', 'v500', 'u850', 'v850', 'u925', 'v925',
    't700', 'pwat',
})


# ── IDX parsing ───────────────────────────────────────────────────────────────

def _parse_idx(idx_text: str,
               patterns: list[tuple[str, str]]) -> dict[str, tuple[int, int | None]]:
    """
    Parse wgrib2 .idx text and return {key: (start_byte, end_byte)} for
    each pattern.  end_byte is None for the last record (read to EOF).

    Format: <rec>:<byte_offset>:<d=YYYYMMDDHH>:<VAR>:<LEVEL>:<TYPE>:
    The first matching line wins (handles duplicates gracefully).
    """
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
    for key, pattern in patterns:
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
      3. Linspace approximation (last resort)
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


# ── Run completeness check ───────────────────────────────────────────────────

async def _run_is_complete(
    client: httpx.AsyncClient, ymd: str, hh: str
) -> bool:
    """
    HEAD-check all 4 required RRFS files concurrently.
    Returns True only if every file returns HTTP 200.
    Uses HEAD (headers only) so no data is transferred.
    """
    prslev_url, prslev_idx_url, twodfd_url, twodfd_idx_url = _source_urls(ymd, hh)
    urls = [prslev_url, prslev_idx_url, twodfd_url, twodfd_idx_url]
    try:
        responses = await asyncio.gather(
            *[client.head(url, timeout=10.0) for url in urls]
        )
        for r, url in zip(responses, urls):
            if r.status_code != 200:
                log.info(f'[rrfs] HEAD {r.status_code}: {url}')
                return False
        return True
    except Exception as e:
        log.warning(f'[rrfs] HEAD check failed: {e}')
        return False


# ── Public entry point ────────────────────────────────────────────────────────

async def fetch_rrfs(cycle_dt: datetime) -> dict | None:
    """
    Fetch RRFS upper-air fields for cycle_dt using HTTP Range requests.

    Uses f000 products (valid_time == run_time).  Tries up to 5 hourly
    runs back starting from cycle_dt rounded to the hour.

    Returns a dict with keys matching the fetch_rap output convention:
      cape, cin, mucape, cape3k, srh1, ustm, vstm,
      u500, v500, u850, v850, u925, v925, u950, v950,
      t700, t925, pwat, lats_rap, lons_rap

    Non-critical fields are set to None rather than aborting the cycle.
    Returns None if no RRFS cycle is available within 5 hours back.
    """
    # Round cycle_dt to the hour (drop sub-hour resolution)
    run_base = cycle_dt.replace(minute=0, second=0, microsecond=0)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for offset in range(5):
            run_dt    = run_base - timedelta(hours=offset)
            ymd       = run_dt.strftime('%Y%m%d')
            hh        = run_dt.strftime('%H')
            run_stamp = run_dt.strftime('%Y%m%d_%H')

            log.info(f'[rrfs] checking run {run_stamp} (offset -{offset}h)')

            # ── HEAD-check all 4 required files before committing to downloads ──
            if not await _run_is_complete(client, ymd, hh):
                log.info(f'[rrfs] {run_stamp}: incomplete — skipping')
                continue

            log.info(f'[rrfs] {run_stamp}: all 4 files present — fetching idx')

            prslev_url, prslev_idx_url, twodfd_url, twodfd_idx_url = (
                _source_urls(ymd, hh)
            )

            # ── Fetch both idx files concurrently ─────────────────────────────
            try:
                prslev_r, twodfd_r = await asyncio.gather(
                    client.get(prslev_idx_url),
                    client.get(twodfd_idx_url),
                )
            except Exception as e:
                log.warning(f'[rrfs] {run_stamp}: idx fetch failed: {e}')
                continue

            try:
                prslev_r.raise_for_status()
                twodfd_r.raise_for_status()
            except Exception as e:
                log.warning(f'[rrfs] {run_stamp}: idx HTTP error: {e}')
                continue

            # ── Parse both idx files ──────────────────────────────────────────
            prslev_ranges = _parse_idx(prslev_r.text, FIELDS_PRSLEV)
            twodfd_ranges = _parse_idx(twodfd_r.text, FIELDS_2DFLD)

            n_found = len(prslev_ranges) + len(twodfd_ranges)
            log.info(f'[rrfs] {run_stamp}: idx parsed — '
                     f'{n_found}/{len(ALL_FIELDS)} fields located '
                     f'(prslev={len(prslev_ranges)}, 2dfld={len(twodfd_ranges)})')

            # Bail early if critical fields are absent from idx
            all_ranges = {**prslev_ranges, **twodfd_ranges}
            missing_critical = CRITICAL_FIELDS - set(all_ranges.keys())
            if missing_critical:
                log.warning(f'[rrfs] {run_stamp}: critical fields absent from idx: '
                             f'{sorted(missing_critical)} — skipping run')
                continue

            # ── Concurrent range fetch + extract ──────────────────────────────
            tasks = []
            for key, (start, end) in prslev_ranges.items():
                tasks.append(
                    _fetch_and_extract(key, start, end, prslev_url, run_stamp, client)
                )
            for key, (start, end) in twodfd_ranges.items():
                tasks.append(
                    _fetch_and_extract(key, start, end, twodfd_url, run_stamp, client)
                )

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

            # Fields not found in idx → ensure None in dict
            for key, _ in ALL_FIELDS:
                data.setdefault(key, None)

            # Check critical fields extracted successfully
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
                     f'{n_loaded}/{len(ALL_FIELDS)} fields loaded; '
                     f'using offset -{offset}h from {run_base.strftime("%Hz")}')
            return data

    log.warning('[rrfs] no RRFS cycle available within 5 hours back')
    return None
