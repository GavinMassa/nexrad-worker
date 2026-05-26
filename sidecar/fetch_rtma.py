import asyncio, logging, tempfile, os
from pathlib import Path
from datetime import datetime, timezone
import httpx
import cfgrib
import numpy as np

log = logging.getLogger(__name__)

NOMADS_BASE = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_rtma2p5.pl'
TMP_DIR = Path(tempfile.gettempdir()) / 'sidecar-cache'
TMP_DIR.mkdir(exist_ok=True)

def rtma_url(cycle_dt: datetime) -> str:
    ymd = cycle_dt.strftime('%Y%m%d')
    hh  = cycle_dt.strftime('%H')
    params = {
        'file':                    f'rtma2p5.t{hh}z.2dvaranl_ndfd.grb2_wexp',
        'var_TMP':                 'on',
        'var_DPT':                 'on',
        'var_UGRD':                'on',
        'var_VGRD':                'on',
        'lev_2_m_above_ground':    'on',
        'lev_10_m_above_ground':   'on',
        'dir':                     f'/rtma2p5.{ymd}',
    }
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    return f'{NOMADS_BASE}?{query}'

async def fetch_rtma(cycle_dt: datetime) -> dict | None:
    """
    Download RTMA 2.5km GRIB2 for cycle_dt, extract surface fields.
    Returns dict with keys: t2m, td2m, u10, v10, lats, lons
    All arrays are float32, shape (ny, nx) for the RTMA CONUS domain.
    Returns None if download fails for both current and previous hour.
    """
    # Try current hour, fall back to previous hour if not available.
    for offset_h in [0, 1]:
        dt = cycle_dt.replace(hour=max(0, cycle_dt.hour - offset_h))
        url  = rtma_url(dt)
        dest = TMP_DIR / f'rtma_{dt.strftime("%Y%m%d_%H")}.grib2'

        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                # HEAD check before committing to a full download.
                head = await client.head(url)
                if head.status_code == 404:
                    log.warning(f'RTMA {dt.strftime("%H")}Z not available (404), trying offset')
                    continue

                log.info(f'Fetching RTMA {dt.strftime("%H")}Z: {url}')
                async with client.stream('GET', url) as r:
                    r.raise_for_status()
                    with open(dest, 'wb') as f:
                        async for chunk in r.aiter_bytes(chunk_size=65536):
                            f.write(chunk)

            log.info(f'RTMA downloaded: {dest.stat().st_size / 1e6:.1f} MB')
            break  # success — stop trying offsets

        except Exception as e:
            log.warning(f'RTMA fetch failed for {dt.strftime("%H")}Z: {e}')
            dest.unlink(missing_ok=True)
            if offset_h == 1:
                return None
    else:
        # Both offsets exhausted without a break.
        return None

    # Extract fields with cfgrib.
    try:
        fields: dict = {}

        # T2m and Td2m — 2 m above ground.
        for short_name, key in [('2t', 't2m'), ('2d', 'td2m')]:
            ds = cfgrib.open_dataset(
                str(dest),
                filter_by_keys={
                    'shortName':   short_name,
                    'typeOfLevel': 'heightAboveGround',
                    'level':       2,
                },
            )
            var_name = list(ds.data_vars)[0]
            fields[key] = ds[var_name].values.astype(np.float32)
            if 'lats' not in fields:
                fields['lats'] = ds['latitude'].values.astype(np.float32)
                fields['lons'] = ds['longitude'].values.astype(np.float32)

        # U10 and V10 — 10 m above ground.
        for short_name, key in [('10u', 'u10'), ('10v', 'v10')]:
            ds = cfgrib.open_dataset(
                str(dest),
                filter_by_keys={
                    'shortName':   short_name,
                    'typeOfLevel': 'heightAboveGround',
                    'level':       10,
                },
            )
            var_name = list(ds.data_vars)[0]
            fields[key] = ds[var_name].values.astype(np.float32)

        ny, nx = fields['t2m'].shape
        log.info(
            f'RTMA extracted: ny={ny} nx={nx} '
            f't2m_sample={fields["t2m"].flat[0]:.1f}K '
            f'td2m_sample={fields["td2m"].flat[0]:.1f}K'
        )

    except Exception as e:
        log.error(f'RTMA cfgrib extraction failed: {e}', exc_info=True)
        dest.unlink(missing_ok=True)
        return None

    # Clean up GRIB file immediately after extraction.
    dest.unlink(missing_ok=True)
    return fields
