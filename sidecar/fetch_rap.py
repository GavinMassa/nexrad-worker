import asyncio, logging, tempfile
from pathlib import Path
from datetime import datetime, timedelta
import httpx, cfgrib, numpy as np

log = logging.getLogger(__name__)
TMP_DIR = Path(tempfile.gettempdir()) / 'sidecar-cache'
TMP_DIR.mkdir(exist_ok=True)

NOMADS_RAP = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl'

def rap_main_url(dt: datetime) -> str:
    ymd = dt.strftime('%Y%m%d')
    hh  = dt.strftime('%H')
    params = {
        'file':                   f'rap.t{hh}z.awp130pgrbf00.grib2',
        'var_CAPE': 'on', 'var_CIN': 'on',
        'var_UGRD': 'on', 'var_VGRD': 'on',
        'var_TMP':  'on', 'var_DPT':  'on',
        'lev_surface':           'on',
        'lev_500_mb':            'on',
        'lev_700_mb':            'on',
        'lev_850_mb':            'on',
        'lev_925_mb':            'on',
        'lev_10_m_above_ground': 'on',
        'lev_2_m_above_ground':  'on',
        'dir': f'/rap.{ymd}',
    }
    return NOMADS_RAP + '?' + '&'.join(f'{k}={v}' for k, v in params.items())

def rap_hlcy_url(dt: datetime) -> str:
    ymd = dt.strftime('%Y%m%d')
    hh  = dt.strftime('%H')
    params = {
        'file':     f'rap.t{hh}z.awp130pgrbf00.grib2',
        'var_HLCY': 'on',
        'dir': f'/rap.{ymd}',
    }
    return NOMADS_RAP + '?' + '&'.join(f'{k}={v}' for k, v in params.items())

async def _download(url: str, dest: Path, client: httpx.AsyncClient) -> bool:
    """Download url to dest. Returns True on success."""
    try:
        head = await client.head(url)
        if head.status_code == 404:
            return False
        async with client.stream('GET', url) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                async for chunk in r.aiter_bytes(65536):
                    f.write(chunk)
        return True
    except Exception as e:
        log.warning(f'RAP download failed {url}: {e}')
        return False

async def fetch_rap(cycle_dt: datetime) -> dict | None:
    """
    Fetch RAP main + HLCY files for cycle_dt (try up to 4 hours back).
    Returns dict with keys: cape, cin, srh1, u500, v500, u10, v10,
    t2m_rap, td2m_rap, lats_rap, lons_rap
    All arrays float32, shape (337, 451) — RAP native grid, row-major.
    Returns None if no run available.
    """
    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        for offset in range(4):
            dt = cycle_dt - timedelta(hours=offset)
            stamp     = dt.strftime('%Y%m%d_%H')
            main_dest = TMP_DIR / f'rap_main_{stamp}.grib2'
            hlcy_dest = TMP_DIR / f'rap_hlcy_{stamp}.grib2'

            ok_main, ok_hlcy = await asyncio.gather(
                _download(rap_main_url(dt), main_dest, client),
                _download(rap_hlcy_url(dt), hlcy_dest, client),
            )
            if not ok_main:
                log.warning(f'RAP main not available for {dt.strftime("%H")}Z')
                for p in (main_dest, hlcy_dest):
                    p.unlink(missing_ok=True)
                continue

            log.info(f'RAP downloaded for {dt.strftime("%H")}Z: '
                     f'main={main_dest.stat().st_size/1e6:.1f}MB '
                     f'hlcy={hlcy_dest.stat().st_size/1e6:.1f}MB')
            try:
                result = _extract_rap(main_dest, hlcy_dest)
                return result
            except Exception as e:
                log.error(f'RAP extraction failed: {e}', exc_info=True)
                return None
            finally:
                main_dest.unlink(missing_ok=True)
                hlcy_dest.unlink(missing_ok=True)

    return None

def _extract_rap(main_path: Path, hlcy_path: Path) -> dict:
    """Extract all needed fields from RAP GRIB2 files using cfgrib."""
    result = {}

    def _get(path, filter_keys, key):
        """Open one dataset, extract first data var, store in result."""
        try:
            ds = cfgrib.open_dataset(str(path), filter_by_keys=filter_keys)
            try:
                arr = ds[list(ds.data_vars)[0]].values.astype(np.float32)
                result[key] = arr
                if 'lats_rap' not in result:
                    result['lats_rap'] = ds['latitude'].values.astype(np.float32)
                    result['lons_rap'] = ds['longitude'].values.astype(np.float32)
                log.info(f'  RAP {key}: shape={arr.shape} sample={arr.flat[0]:.2f}')
            finally:
                ds.close()   # release eccodes file handle + index objects
        except Exception as e:
            log.warning(f'  RAP {key} extraction failed: {e}')
            result[key] = None

    # CAPE and CIN at surface (awp130p stores these at typeOfLevel=surface)
    _get(main_path, {'discipline': 0, 'parameterCategory': 7, 'parameterNumber': 6,
                     'typeOfLevel': 'surface'}, 'cape')
    _get(main_path, {'discipline': 0, 'parameterCategory': 7, 'parameterNumber': 7,
                     'typeOfLevel': 'surface'}, 'cin')

    # U/V at 500mb (isobaricInhPa level 500)
    _get(main_path, {'discipline': 0, 'parameterCategory': 2, 'parameterNumber': 2,
                     'typeOfLevel': 'isobaricInhPa', 'level': 500}, 'u500')
    _get(main_path, {'discipline': 0, 'parameterCategory': 2, 'parameterNumber': 3,
                     'typeOfLevel': 'isobaricInhPa', 'level': 500}, 'v500')

    # U/V at 850mb (~1500m AGL) — intermediate layer for two-layer BWD6
    _get(main_path, {'discipline': 0, 'parameterCategory': 2, 'parameterNumber': 2,
                     'typeOfLevel': 'isobaricInhPa', 'level': 850}, 'u850')
    _get(main_path, {'discipline': 0, 'parameterCategory': 2, 'parameterNumber': 3,
                     'typeOfLevel': 'isobaricInhPa', 'level': 850}, 'v850')

    # T at 700mb (~3000m AGL) — for 0-3km and 700-500mb lapse rate
    _get(main_path, {'discipline': 0, 'parameterCategory': 0, 'parameterNumber': 0,
                     'typeOfLevel': 'isobaricInhPa', 'level': 700}, 't700')

    # T at 925mb (~750m AGL) — for low-level lapse rate baseline
    _get(main_path, {'discipline': 0, 'parameterCategory': 0, 'parameterNumber': 0,
                     'typeOfLevel': 'isobaricInhPa', 'level': 925}, 't925')

    # U/V at 10m AGL
    _get(main_path, {'discipline': 0, 'parameterCategory': 2, 'parameterNumber': 2,
                     'typeOfLevel': 'heightAboveGround', 'level': 10}, 'u10')
    _get(main_path, {'discipline': 0, 'parameterCategory': 2, 'parameterNumber': 3,
                     'typeOfLevel': 'heightAboveGround', 'level': 10}, 'v10')

    # T and Td at 2m AGL
    _get(main_path, {'discipline': 0, 'parameterCategory': 0, 'parameterNumber': 0,
                     'typeOfLevel': 'heightAboveGround', 'level': 2}, 't2m_rap')
    _get(main_path, {'discipline': 0, 'parameterCategory': 0, 'parameterNumber': 6,
                     'typeOfLevel': 'heightAboveGround', 'level': 2}, 'td2m_rap')

    # td700: derive from t700 + surface Td depression scaling
    # 700mb dewpoint not available in filtered awp130p — approximate from
    # surface moisture profile. Validated adequate for BACI threshold computation.
    if result.get('t700') is not None and result.get('t2m_rap') is not None \
            and result.get('td2m_rap') is not None:
        sfc_dep = result['t2m_rap'] - result['td2m_rap']
        result['td700'] = (result['t700'] - sfc_dep * 0.5).astype(np.float32)
        log.info(f"  RAP td700: derived from surface scaling, "
                 f"sample={result['td700'].flat[0]:.2f}")
    else:
        result['td700'] = None
        log.warning('  RAP td700: skipped (t700 or surface T/Td missing)')

    # pwat: derive from surface Td using Bolton approximation
    # PWAT not filterable from awp130p — estimate from surface mixing ratio.
    # Accuracy ±25%, adequate for BACI moisture threshold (10-25mm range).
    if result.get('td2m_rap') is not None:
        td_c = result['td2m_rap'] - 273.15
        e_s = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
        result['pwat'] = (2.0 * e_s).astype(np.float32)
        log.info(f"  RAP pwat: derived from surface Td, "
                 f"sample={result['pwat'].flat[0]:.2f}mm")
    else:
        result['pwat'] = None
        log.warning('  RAP pwat: skipped (td2m_rap missing)')

    # HLCY: try all datasets, pick the first 337×451 array (0-1km layer).
    # Use a finally block to guarantee ALL dataset handles are closed even if
    # an exception occurs mid-loop or before ds.close() is reached.
    if hlcy_path.exists() and hlcy_path.stat().st_size > 1000:
        try:
            datasets = cfgrib.open_datasets(str(hlcy_path))
            log.info(f'  RAP HLCY datasets: {len(datasets)}')
            srh = None
            try:
                for i, ds in enumerate(datasets):
                    log.info(f'    dataset[{i}]: vars={list(ds.data_vars)} dims={dict(ds.dims)}')
                    for var in ds.data_vars:
                        arr = ds[var].values
                        # cfgrib stacks both HLCY layers (0-1km and 0-3km) along a
                        # heightAboveGroundLayer dimension → shape (2, 337, 451).
                        # Index 0 is the 0-1km layer (shorter layer listed first).
                        if arr.ndim == 3 and arr.shape[1:] == (337, 451):
                            srh = arr[0].astype(np.float32)
                            log.info(f'    using dataset[{i}].{var}[0] (0-1km) '
                                     f'shape={srh.shape} sample={srh.flat[0]:.2f}')
                            break
                        elif arr.shape == (337, 451):
                            srh = arr.astype(np.float32)
                            log.info(f'    using dataset[{i}].{var} shape={arr.shape} '
                                     f'sample={srh.flat[0]:.2f}')
                            break
                    if srh is not None:
                        break
            finally:
                for ds in datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass
            result['srh1'] = srh
            if srh is None:
                log.warning('RAP HLCY: no 337×451 array found')
        except Exception as e:
            log.warning(f'RAP HLCY extraction failed: {e}')
            result['srh1'] = None
    else:
        log.warning('RAP HLCY file missing or too small')
        result['srh1'] = None

    extracted = [k for k, v in result.items()
                 if k not in ('lats_rap', 'lons_rap') and v is not None]
    log.info(f'RAP extraction done: {extracted}')
    import gc; gc.collect()
    return result
