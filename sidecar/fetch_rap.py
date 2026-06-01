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
        'lev_950_mb':            'on',
        'lev_10_m_above_ground': 'on',
        'lev_2_m_above_ground':  'on',
        'dir': f'/rap.{ymd}',
    }
    return NOMADS_RAP + '?' + '&'.join(f'{k}={v}' for k, v in params.items())

def rap_hlcy_url(dt: datetime) -> str:
    ymd = dt.strftime('%Y%m%d')
    hh  = dt.strftime('%H')
    params = {
        'file':                      f'rap.t{hh}z.awp130pgrbf00.grib2',
        'var_HLCY':                  'on',
        'lev_3000-0_m_above_ground': 'on',
        'lev_1000-0_m_above_ground': 'on',
        'dir': f'/rap.{ymd}',
    }
    return NOMADS_RAP + '?' + '&'.join(f'{k}={v}' for k, v in params.items())

def rap_stm_url(dt: datetime) -> str:
    """USTM/VSTM storm motion — fetched separately, optional."""
    ymd = dt.strftime('%Y%m%d')
    hh  = dt.strftime('%H')
    params = {
        'file':                      f'rap.t{hh}z.awp130pgrbf00.grib2',
        'var_USTM':                  'on',
        'var_VSTM':                  'on',
        'lev_0-6000_m_above_ground': 'on',
        'dir': f'/rap.{ymd}',
    }
    return NOMADS_RAP + '?' + '&'.join(f'{k}={v}' for k, v in params.items())

def rap_pwat_url(dt: datetime) -> str:
    """PWAT precipitable water — fetched separately, optional."""
    ymd = dt.strftime('%Y%m%d')
    hh  = dt.strftime('%H')
    params = {
        'file':     f'rap.t{hh}z.awp130pgrbf00.grib2',
        'var_PWAT': 'on',
        'lev_entire_atmosphere_(considered_as_a_single_layer)': 'on',
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
    Fetch RAP main + HLCY + optional STM/PWAT files for cycle_dt
    (tries up to 4 hours back). STM and PWAT are optional — a 500 from
    NOMADS on those URLs does not abort the cycle.
    Returns dict with keys: cape, cin, srh1, u500, v500, u10, v10,
    t2m_rap, td2m_rap, lats_rap, lons_rap, [ustm, vstm, pwat]
    All arrays float32, shape (337, 451) — RAP native grid, row-major.
    Returns None if no run available.
    """
    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        for offset in range(4):
            dt = cycle_dt - timedelta(hours=offset)
            stamp     = dt.strftime('%Y%m%d_%H')
            main_dest = TMP_DIR / f'rap_main_{stamp}.grib2'
            hlcy_dest = TMP_DIR / f'rap_hlcy_{stamp}.grib2'
            stm_dest  = TMP_DIR / f'rap_stm_{stamp}.grib2'
            pwat_dest = TMP_DIR / f'rap_pwat_{stamp}.grib2'

            ok_main, ok_hlcy, ok_stm, ok_pwat = await asyncio.gather(
                _download(rap_main_url(dt),  main_dest, client),
                _download(rap_hlcy_url(dt),  hlcy_dest, client),
                _download(rap_stm_url(dt),   stm_dest,  client),
                _download(rap_pwat_url(dt),  pwat_dest, client),
            )
            if not ok_main:
                log.warning(f'RAP main not available for {dt.strftime("%H")}Z')
                for p in (main_dest, hlcy_dest, stm_dest, pwat_dest):
                    p.unlink(missing_ok=True)
                continue

            log.info(
                f'RAP downloaded for {dt.strftime("%H")}Z: '
                f'main={main_dest.stat().st_size/1e6:.1f}MB '
                f'hlcy={"%.1fMB" % (hlcy_dest.stat().st_size/1e6) if ok_hlcy else "N/A"} '
                f'stm={"ok" if ok_stm else "N/A"} '
                f'pwat={"ok" if ok_pwat else "N/A"}'
            )
            try:
                result = _extract_rap(
                    main_dest,
                    hlcy_dest if ok_hlcy else None,
                    stm_dest  if ok_stm  else None,
                    pwat_dest if ok_pwat else None,
                )
                return result
            except Exception as e:
                log.error(f'RAP extraction failed: {e}', exc_info=True)
                return None
            finally:
                for p in (main_dest, hlcy_dest, stm_dest, pwat_dest):
                    p.unlink(missing_ok=True)

    return None

def _extract_rap(main_path: Path,
                 hlcy_path: Path | None,
                 stm_path:  Path | None = None,
                 pwat_path: Path | None = None) -> dict:
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

    # ── Main file ─────────────────────────────────────────────────────────────

    # CAPE and CIN at surface (awp130p stores these at typeOfLevel=surface)
    _get(main_path, {'discipline': 0, 'parameterCategory': 7, 'parameterNumber': 6,
                     'typeOfLevel': 'surface'}, 'cape')
    _get(main_path, {'discipline': 0, 'parameterCategory': 7, 'parameterNumber': 7,
                     'typeOfLevel': 'surface'}, 'cin')

    # ── Isobaric U/V winds — all pressure levels in one cfgrib open ──────────
    # Filtering once by typeOfLevel+shortName avoids 8 separate file opens and
    # prevents mismatched u/v if any individual per-level filter fails.
    # 950mb requires lev_950_mb=on in rap_main_url().
    _iso_levels = {500: ('u500', 'v500'), 850: ('u850', 'v850'),
                   925: ('u925', 'v925'), 950: ('u950', 'v950')}
    for _k in [k for pair in _iso_levels.values() for k in pair]:
        result[_k] = None   # pre-fill; overwritten per level on success

    try:
        _ds_iso = cfgrib.open_dataset(str(main_path), filter_by_keys={
            'typeOfLevel': 'isobaricInhPa',
            'shortName':   ['u', 'v'],
        })
        try:
            _lev_dim = 'isobaricInhPa' if 'isobaricInhPa' in _ds_iso.coords else 'level'
            for _lvl, (_ku, _kv) in _iso_levels.items():
                try:
                    _u = _ds_iso['u'].sel({_lev_dim: _lvl}).values.astype(np.float32)
                    _v = _ds_iso['v'].sel({_lev_dim: _lvl}).values.astype(np.float32)
                    result[_ku] = _u
                    result[_kv] = _v
                    if 'lats_rap' not in result:
                        result['lats_rap'] = _ds_iso['latitude'].values.astype(np.float32)
                        result['lons_rap']  = _ds_iso['longitude'].values.astype(np.float32)
                    log.info(f'  RAP {_ku}: shape={_u.shape} sample={_u.flat[0]:.2f}')
                    log.info(f'  RAP {_kv}: shape={_v.shape} sample={_v.flat[0]:.2f}')
                except Exception as _e:
                    log.warning(f'  RAP {_ku}/{_kv} slice failed: {_e}')
        finally:
            _ds_iso.close()
    except Exception as e:
        log.warning(f'  RAP isobaric wind open failed — all levels set to None: {e}')

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

    # ── Derived fields ────────────────────────────────────────────────────────

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

    # ── HLCY file ─────────────────────────────────────────────────────────────
    # Try all datasets, pick the first 337×451 array (0-1km layer).
    # Use a finally block to guarantee ALL dataset handles are closed even if
    # an exception occurs mid-loop or before ds.close() is reached.
    if hlcy_path is not None and hlcy_path.exists() and hlcy_path.stat().st_size > 1000:
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

    # ── STM file (optional) — USTM/VSTM storm motion ─────────────────────────
    if stm_path and stm_path.exists():
        try:
            import xarray as xr
            ds_motion = xr.open_dataset(str(stm_path), engine='cfgrib')
            try:
                result['ustm'] = ds_motion['ustm'].values.astype(np.float32)
                result['vstm'] = ds_motion['vstm'].values.astype(np.float32)
                log.info(f'  RAP ustm: shape={result["ustm"].shape} sample={result["ustm"].flat[0]:.2f}')
                log.info(f'  RAP vstm: shape={result["vstm"].shape} sample={result["vstm"].flat[0]:.2f}')
            finally:
                ds_motion.close()
        except Exception as e:
            log.warning(f'  RAP ustm/vstm extraction failed: {e}')
            result['ustm'] = None
            result['vstm'] = None
    else:
        result['ustm'] = None
        result['vstm'] = None
        if stm_path is not None:
            log.warning('RAP STM file missing or too small — ustm/vstm unavailable')

    # ── PWAT file (optional) — precipitable water ─────────────────────────────
    # Use real GRIB2 field when available; fall back to Bolton approximation.
    pwat_loaded = False
    if pwat_path is not None and pwat_path.exists() and pwat_path.stat().st_size > 1000:
        _get(pwat_path, {'discipline': 0, 'parameterCategory': 1, 'parameterNumber': 3,
                         'typeOfLevel': 'atmosphereSingleLayer'}, 'pwat_real')
        if result.get('pwat_real') is not None:
            result['pwat'] = result.pop('pwat_real')
            log.info(f"  RAP pwat: real GRIB2 field, "
                     f"sample={result['pwat'].flat[0]:.2f}kg/m^2")
            pwat_loaded = True
        else:
            log.warning('RAP PWAT file present but extraction failed — falling back to Bolton')

    if not pwat_loaded:
        if result.get('td2m_rap') is not None:
            td_c = result['td2m_rap'] - 273.15
            e_s = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
            result['pwat'] = (2.0 * e_s).astype(np.float32)
            log.info(f"  RAP pwat: derived from surface Td (PWAT field unavailable), "
                     f"sample={result['pwat'].flat[0]:.2f}mm")
        else:
            result['pwat'] = None
            log.warning('  RAP pwat: skipped (td2m_rap missing)')

    # ── Summary ───────────────────────────────────────────────────────────────
    extracted = [k for k, v in result.items()
                 if k not in ('lats_rap', 'lons_rap') and v is not None]
    log.info(f'RAP extraction done: {extracted}')
    import gc; gc.collect()
    return result
