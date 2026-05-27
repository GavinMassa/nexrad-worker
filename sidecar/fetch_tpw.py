import asyncio, logging, re, tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import httpx

log = logging.getLogger(__name__)
TMP_DIR = Path(tempfile.gettempdir()) / 'sidecar-cache'
TMP_DIR.mkdir(exist_ok=True)

# GOES-18 ABI L2 TPW CONUS sector — public AWS S3, no auth required.
# Full sector (~5-8 MB per file) at 2km resolution, updated every ~10 min.
GOES18_S3_BASE = 'https://noaa-goes18.s3.amazonaws.com'


async def fetch_latest_tpw(now: datetime) -> dict | None:
    """
    Fetch the most recent GOES-18 ABI L2 TPWF (CONUS) file for the hour
    indicated by `now`, falling back to the previous hour if nothing is found.

    Returns dict: {'tpw': float32 (ny,nx) in mm,
                   'lats': float32 (ny,nx),
                   'lons': float32 (ny,nx)}
    Shape is typically (1500, 2500) for the GOES-18 CONUS sector at 2km.
    Returns None on any failure — never raises.
    """
    for offset in (0, 1):
        dt   = now - timedelta(hours=offset)
        year = dt.strftime('%Y')
        doy  = dt.strftime('%j')
        hour = dt.strftime('%H')
        prefix   = f'ABI-L2-TPWF/{year}/{doy}/{hour}/'
        list_url = f'{GOES18_S3_BASE}?list-type=2&prefix={prefix}&max-keys=10'

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(list_url)
                r.raise_for_status()
                keys = re.findall(r'(ABI-L2-TPWF[^<]+\.nc)', r.text)
        except Exception as e:
            log.warning(f'[tpw] S3 listing failed (offset={offset}h): {e}')
            continue

        if not keys:
            log.info(f'[tpw] no files in {prefix}, trying previous hour')
            continue

        # Use the last key — S3 lists in alphabetical order, last = most recent scan.
        key      = keys[-1]
        dest     = TMP_DIR / Path(key).name
        file_url = f'{GOES18_S3_BASE}/{key}'

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream('GET', file_url) as r:
                    r.raise_for_status()
                    with open(dest, 'wb') as f:
                        async for chunk in r.aiter_bytes(65536):
                            f.write(chunk)
            log.info(f'[tpw] downloaded {key}: {dest.stat().st_size/1e6:.1f}MB')
        except Exception as e:
            log.warning(f'[tpw] download failed: {e}')
            dest.unlink(missing_ok=True)
            continue

        try:
            result = _extract_tpw(dest)
            return result
        except Exception as e:
            log.warning(f'[tpw] extraction failed: {e}')
            return None
        finally:
            dest.unlink(missing_ok=True)

    log.warning('[tpw] no TPW file found for current or previous hour')
    return None


def _extract_tpw(nc_path: Path) -> dict | None:
    """
    Extract TPW and compute lat/lon arrays from a GOES-18 ABI L2 TPWF netCDF4 file.

    Uses the GOES-R ABI fixed-grid projection formulas (NOAA PUG Vol 3 §4.2.8.1)
    to convert scan-angle (x, y) coordinates to geodetic (lat, lon).

    Returns dict: {'tpw': float32 (ny,nx) mm,
                   'lats': float32 (ny,nx),
                   'lons': float32 (ny,nx)}
    Returns None if netCDF4 is unavailable or the file cannot be parsed.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        log.error('[tpw] netCDF4 not installed — add to sidecar/requirements.txt')
        return None

    with nc.Dataset(str(nc_path), 'r') as ds:
        tpw = ds.variables['TPW'][:].astype(np.float32)   # (y, x) mm
        x   = ds.variables['x'][:].astype(np.float64)     # scan angle, radians
        y   = ds.variables['y'][:].astype(np.float64)
        proj = ds.variables['goes_imager_projection']
        # Satellite height above Earth centre (perspective point height + r_eq)
        H     = float(proj.perspective_point_height) + 6378137.0
        r_eq  = float(proj.semi_major_axis)
        r_pol = float(proj.semi_minor_axis)
        lon_0 = float(proj.longitude_of_projection_origin) * np.pi / 180.0

    # GOES-R ABI fixed-grid → geodetic lat/lon (NOAA PUG Vol 3 §4.2.8.1)
    X, Y = np.meshgrid(x, y)
    cos_x, sin_x = np.cos(X), np.sin(X)
    cos_y, sin_y = np.cos(Y), np.sin(Y)

    a = sin_x**2 + cos_x**2 * (cos_y**2 + (r_eq / r_pol)**2 * sin_y**2)
    b = -2.0 * H * cos_x * cos_y
    c = H**2 - r_eq**2

    discriminant = np.maximum(b**2 - 4.0 * a * c, 0.0)
    rs = (-b - np.sqrt(discriminant)) / (2.0 * a)

    sx = rs * cos_x * cos_y
    sy = -rs * sin_x
    sz = rs * cos_x * sin_y

    lats = np.degrees(np.arctan((r_eq / r_pol)**2 * sz / np.sqrt((H - sx)**2 + sy**2)))
    lons = np.degrees(lon_0 - np.arctan(sy / (H - sx)))

    # Mask fill values — GOES L2 TPW uses DQF==0 for good data, but the
    # simplest filter is to zero out the known netCDF4 fill value (typically
    # 65535 for uint16, or the float _FillValue if scaled).
    # Use a high-value threshold that excludes any physically realistic TPW.
    tpw = np.where(tpw > 200.0, np.nan, tpw)   # TPW never exceeds ~180mm globally
    tpw = np.where(tpw < 0.0,   np.nan, tpw)

    n_valid = int(np.isfinite(tpw).sum())
    log.info(f'[tpw] extracted: shape={tpw.shape} valid={n_valid} '
             f'tpw_max={float(np.nanmax(tpw)) if n_valid > 0 else 0:.1f}mm')

    return {
        'tpw':  tpw.astype(np.float32),
        'lats': lats.astype(np.float32),
        'lons': lons.astype(np.float32),
    }
