import logging
import os
import numpy as np
import httpx
import psutil
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# IEM ASOS/mesonet current obs endpoint.
# Returns all US stations with obs within the last 90 minutes.
IEM_CURRENT_URL = 'https://mesonet.agron.iastate.edu/api/1/currents.geojson'

# IDW search radius in degrees (~300 km). Stations outside this radius
# from a given gridpoint contribute negligible weight.
IDW_RADIUS_DEG = 2.5

# IDW power parameter. p=2 gives inverse-square weighting — standard for
# meteorological objective analysis at this scale.
IDW_POWER = 2.0

# Minimum number of stations required within IDW_RADIUS_DEG to apply a
# correction at a gridpoint. Gridpoints with fewer nearby stations keep
# the raw RTMA value (innovation = 0).
MIN_STATIONS = 3

# Maximum plausible innovation magnitude (K). Observations differing from
# RTMA by more than this are quality-controlled out — likely bad sensors.
MAX_INNOVATION_K = 8.0


async def fetch_mesonet_obs() -> list[dict] | None:
    """
    Fetch current surface obs from IEM.
    Returns list of dicts with keys: lat, lon, t_k, td_k (all in Kelvin).
    Returns None on network failure or if no valid stations found.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(IEM_CURRENT_URL)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning(f'[mesonet] fetch failed: {e}')
        return None

    stations = []
    for feature in data.get('features', []):
        props  = feature.get('properties', {})
        geom   = feature.get('geometry',   {})
        coords = geom.get('coordinates')
        if not coords or len(coords) < 2:
            continue
        tmpf = props.get('tmpf')
        dwpf = props.get('dwpf')
        if tmpf is None or dwpf is None:
            continue
        try:
            stations.append({
                'lat':  float(coords[1]),
                'lon':  float(coords[0]),
                't_k':  (float(tmpf) - 32.0) * 5.0 / 9.0 + 273.15,
                'td_k': (float(dwpf) - 32.0) * 5.0 / 9.0 + 273.15,
            })
        except (TypeError, ValueError):
            continue

    log.info(f'[mesonet] {len(stations)} stations with valid T/Td')
    return stations if stations else None


def _nearest_rtma_values(
    t2m:     np.ndarray,   # (ny, nx)
    td2m:    np.ndarray,   # (ny, nx)
    lats:    np.ndarray,   # (ny, nx)
    lons:    np.ndarray,   # (ny, nx) in -180..180
    st_lats: np.ndarray,   # (n_st,)
    st_lons: np.ndarray,   # (n_st,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each station find the nearest RTMA gridpoint using KDTree (fast for >1000 stations).
    Returns (st_rtma_t, st_rtma_td): float32 arrays shape (n_st,).
    """
    ny, nx = t2m.shape
    grid_pts = np.column_stack([lats.ravel(), lons.ravel()])   # (ny*nx, 2)
    tree     = KDTree(grid_pts)

    st_pts   = np.column_stack([st_lats, st_lons])             # (n_st, 2)
    _, idx   = tree.query(st_pts)                               # (n_st,)

    iy, ix = np.unravel_index(idx, (ny, nx))
    return t2m[iy, ix].astype(np.float32), td2m[iy, ix].astype(np.float32)


def compute_innovation_field(
    rtma_t2m:   np.ndarray,   # (ny, nx) K  — must be a small (downsampled) grid
    rtma_td2m:  np.ndarray,   # (ny, nx) K
    rtma_lats:  np.ndarray,   # (ny, nx)
    rtma_lons:  np.ndarray,   # (ny, nx) already in -180..180
    stations:   list[dict],
    st_rtma_t:  np.ndarray,   # (n_st,) RTMA T at each station's nearest gridpoint
    st_rtma_td: np.ndarray,   # (n_st,) RTMA Td at each station's nearest gridpoint
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IDW innovation fields (delta_t, delta_td) on the passed-in grid.

    The caller is responsible for passing a grid small enough that a full
    (n_gp × n_stations) float32 distance matrix fits in RAM.  At the
    downsampled resolution (399×586 ≈ 234k points, ~8800 stations):
        234k × 8800 × 4 bytes ≈ 8 MB   ← safe, no chunking needed

    For each gridpoint:
      1. Find all stations within IDW_RADIUS_DEG
      2. Compute obs - rtma_background at each station (the innovation)
      3. QC: discard innovations outside ±MAX_INNOVATION_K
      4. Spread innovations to gridpoint using IDW weights
      5. If < MIN_STATIONS pass QC, innovation = 0 (keep RTMA)

    Returns (delta_t, delta_td): float32 arrays shape (ny, nx).
    """
    ny, nx = rtma_t2m.shape

    if not stations:
        return np.zeros((ny, nx), dtype=np.float32), np.zeros((ny, nx), dtype=np.float32)

    st_lats = np.array([s['lat']  for s in stations], dtype=np.float32)
    st_lons = np.array([s['lon']  for s in stations], dtype=np.float32)
    st_t    = np.array([s['t_k']  for s in stations], dtype=np.float32)
    st_td   = np.array([s['td_k'] for s in stations], dtype=np.float32)

    # Obs-minus-background innovations — constant per station across all gridpoints
    innov_t  = st_t  - st_rtma_t    # (n_st,)
    innov_td = st_td - st_rtma_td

    # QC: discard stations with physically implausible innovations
    qc_ok = (
        (np.abs(innov_t)  < MAX_INNOVATION_K) &
        (np.abs(innov_td) < MAX_INNOVATION_K)
    )
    n_qc_fail = int((~qc_ok).sum())
    if n_qc_fail:
        log.info(f'[mesonet] QC removed {n_qc_fail} stations '
                 f'with |innovation| > {MAX_INNOVATION_K} K')

    # Full distance matrix — safe because caller passes a small grid
    gp_lats = rtma_lats.ravel()   # (n_gp,)
    gp_lons = rtma_lons.ravel()

    dlat = gp_lats[:, None] - st_lats[None, :]   # (n_gp, n_st)
    dlon = gp_lons[:, None] - st_lons[None, :]
    dist = np.sqrt(dlat**2 + dlon**2)             # equirectangular, degrees

    weights    = (1.0 / np.maximum(dist, 1e-6) ** IDW_POWER) * (dist < IDW_RADIUS_DEG)
    weights_qc = weights * qc_ok[None, :]         # zero out QC-failed stations

    n_valid = (weights_qc > 0).sum(axis=1)        # (n_gp,)
    w_sum   = weights_qc.sum(axis=1)
    w_safe  = np.where(w_sum > 0, w_sum, 1.0)

    idw_t  = (weights_qc * innov_t [None, :]).sum(axis=1) / w_safe
    idw_td = (weights_qc * innov_td[None, :]).sum(axis=1) / w_safe

    mask     = (n_valid >= MIN_STATIONS).reshape(ny, nx)
    delta_t  = np.where(mask, idw_t .reshape(ny, nx), 0.0).astype(np.float32)
    delta_td = np.where(mask, idw_td.reshape(ny, nx), 0.0).astype(np.float32)
    return delta_t, delta_td


def inject_observations(
    rtma:     dict,
    stations: list[dict],
) -> dict:
    """
    Apply mesonet IDW corrections to RTMA T/Td. Returns a new dict with
    corrected 't2m' and 'td2m'; all other keys are shared references (not copied).

    Called in main.py between fetch_rtma() and blend().
    Mesonet failure must never crash the cycle — the caller wraps this in try/except.
    """
    ny, nx = rtma['t2m'].shape
    log.info(f'[mesonet] inject_observations called with grid {ny}×{nx}')
    assert ny < 500 and nx < 700, (
        f'Grid too large for mesonet injection: {ny}×{nx} — must be downsampled first'
    )

    if not stations:
        log.info('[mesonet] no stations — skipping injection, using raw RTMA')
        return rtma

    t2m      = rtma['t2m']
    td2m     = rtma['td2m']
    lats     = rtma['lats']
    lons_raw = rtma['lons']
    # Normalise to -180..180 for distance calc (same convention as writer.py)
    lons     = np.where(lons_raw > 180.0, lons_raw - 360.0, lons_raw)

    # Filter to RTMA domain bbox — skip Alaska, Hawaii, overseas stations
    lat_min_g = float(lats.min());  lat_max_g = float(lats.max())
    lon_min_g = float(lons.min());  lon_max_g = float(lons.max())

    stations_conus = [
        s for s in stations
        if lat_min_g <= s['lat'] <= lat_max_g
        and lon_min_g <= s['lon'] <= lon_max_g
    ]
    log.info(f'[mesonet] {len(stations_conus)}/{len(stations)} stations within RTMA domain')

    if not stations_conus:
        log.warning('[mesonet] no in-domain stations — using raw RTMA')
        return rtma

    st_lats = np.array([s['lat'] for s in stations_conus], dtype=np.float32)
    st_lons = np.array([s['lon'] for s in stations_conus], dtype=np.float32)

    # Nearest RTMA gridpoint value at each station (KDTree — O(n log N) vs O(n·N))
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    log.info(f'[mesonet] memory before KDTree: {mem:.2f} GB')
    log.info(f'[mesonet] KDTree lookup for {len(stations_conus)} stations on '
             f'{t2m.shape[0]}×{t2m.shape[1]} grid...')
    st_rtma_t, st_rtma_td = _nearest_rtma_values(
        t2m, td2m, lats, lons, st_lats, st_lons
    )

    log.info('[mesonet] computing IDW innovation field...')
    delta_t, delta_td = compute_innovation_field(
        t2m, td2m, lats, lons,
        stations_conus, st_rtma_t, st_rtma_td,
    )

    n_corrected = int((np.abs(delta_t) > 0.01).sum())
    t_max  = float(np.abs(delta_t).max())
    td_max = float(np.abs(delta_td).max())
    log.info(f'[mesonet] {n_corrected} gridpoints corrected, '
             f'max |ΔT|={t_max:.2f} K  max |ΔTd|={td_max:.2f} K')

    corrected = dict(rtma)                                      # shallow copy
    corrected['t2m']  = (t2m  + delta_t ).astype(np.float32)
    corrected['td2m'] = (td2m + delta_td).astype(np.float32)
    return corrected
