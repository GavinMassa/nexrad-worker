import logging
import numpy as np
import httpx
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# IEM ASOS/mesonet current obs endpoint.
# Returns all US stations with obs within the last 90 minutes.
IEM_CURRENT_URL = 'https://mesonet.agron.iastate.edu/api/1/currents.geojson'

# IDW search radius in degrees (~300 km).
IDW_RADIUS_DEG = 2.5

# IDW power parameter. p=2 gives inverse-square weighting.
IDW_POWER = 2.0

# Minimum stations within IDW_RADIUS_DEG before a gridpoint is corrected.
MIN_STATIONS = 3

# Maximum plausible innovation magnitude (K). Larger departures = bad sensor.
MAX_INNOVATION_K = 8.0


async def fetch_mesonet_obs() -> list[dict] | None:
    """
    Fetch current surface obs from IEM.
    Returns list of dicts with keys: lat, lon, t_k, td_k (Kelvin).
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


def spatial_thin(stations: list[dict], grid_spacing_deg: float = 0.45) -> list[dict]:
    """
    Retain at most one station per grid_spacing_deg × grid_spacing_deg cell.
    For each occupied cell, keep the station closest to the cell centre.

    0.45° ≈ 50 km — reduces 8000+ raw stations to ~500-800 over CONUS,
    adequate for 10 km IDW and keeps the distance matrix under ~560 MB worst
    case (233k gridpoints × 800 stations × 4 bytes).
    """
    if not stations:
        return []
    cells: dict[tuple, dict] = {}
    for s in stations:
        cell = (round(s['lat'] / grid_spacing_deg),
                round(s['lon'] / grid_spacing_deg))
        if cell not in cells:
            cells[cell] = s
        else:
            # Keep whichever station is closer to the cell centre
            clat = cell[0] * grid_spacing_deg
            clon = cell[1] * grid_spacing_deg
            cur  = cells[cell]
            d_cur = (cur['lat'] - clat) ** 2 + (cur['lon'] - clon) ** 2
            d_new = (s['lat']   - clat) ** 2 + (s['lon']   - clon) ** 2
            if d_new < d_cur:
                cells[cell] = s
    thinned = list(cells.values())
    log.info(f'[mesonet] thinned {len(stations)} → {len(thinned)} stations '
             f'(grid_spacing={grid_spacing_deg}°)')
    return thinned


def compute_correction(
    stations:   list[dict],
    grid_lats:  np.ndarray,   # (ny, nx) — downsampled RTMA lats
    grid_lons:  np.ndarray,   # (ny, nx) — downsampled RTMA lons, -180..180
    grid_t2m:   np.ndarray,   # (ny, nx) K — background temperature
    grid_td2m:  np.ndarray,   # (ny, nx) K — background dewpoint
) -> tuple[np.ndarray, np.ndarray]:
    """
    IDW innovation field on a small (downsampled) grid.

    The caller passes the 399×586 blend-output grid. At that resolution with
    ~600 thinned stations the full distance matrix is:
        233k × 600 × 4 bytes ≈ 560 MB worst case, ~280 MB typical.
    Safe to allocate without chunking.

    Returns (delta_t, delta_td): float32 arrays shape (ny, nx).
    Add to RTMA T/Td before blend() to apply the obs correction.
    """
    ny, nx = grid_t2m.shape

    if not stations:
        return np.zeros((ny, nx), np.float32), np.zeros((ny, nx), np.float32)

    st_lats = np.array([s['lat']  for s in stations], dtype=np.float32)
    st_lons = np.array([s['lon']  for s in stations], dtype=np.float32)
    st_t    = np.array([s['t_k']  for s in stations], dtype=np.float32)
    st_td   = np.array([s['td_k'] for s in stations], dtype=np.float32)

    # Nearest gridpoint background value at each station via KDTree
    gp_pts = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])  # (n_gp, 2)
    tree   = KDTree(gp_pts)
    st_pts = np.column_stack([st_lats, st_lons])                       # (n_st, 2)
    _, idx = tree.query(st_pts)
    iy, ix = np.unravel_index(idx, (ny, nx))
    st_bg_t  = grid_t2m [iy, ix].astype(np.float32)
    st_bg_td = grid_td2m[iy, ix].astype(np.float32)

    innov_t  = st_t  - st_bg_t    # obs-minus-background (n_st,)
    innov_td = st_td - st_bg_td

    # QC: discard stations with implausible innovations
    qc_ok = (
        (np.abs(innov_t)  < MAX_INNOVATION_K) &
        (np.abs(innov_td) < MAX_INNOVATION_K)
    )
    n_bad = int((~qc_ok).sum())
    if n_bad:
        log.info(f'[mesonet] QC removed {n_bad} stations')

    # Full distance matrix — safe at downsampled grid resolution
    gp_lats = grid_lats.ravel()
    gp_lons = grid_lons.ravel()

    dlat = gp_lats[:, None] - st_lats[None, :]   # (n_gp, n_st)
    dlon = gp_lons[:, None] - st_lons[None, :]
    dist = np.sqrt(dlat ** 2 + dlon ** 2)

    weights    = (1.0 / np.maximum(dist, 1e-6) ** IDW_POWER) * (dist < IDW_RADIUS_DEG)
    weights_qc = weights * qc_ok[None, :]

    n_valid = (weights_qc > 0).sum(axis=1)        # (n_gp,)
    w_sum   = weights_qc.sum(axis=1)
    w_safe  = np.where(w_sum > 0, w_sum, 1.0)

    idw_t  = (weights_qc * innov_t [None, :]).sum(axis=1) / w_safe
    idw_td = (weights_qc * innov_td[None, :]).sum(axis=1) / w_safe

    mask     = (n_valid >= MIN_STATIONS).reshape(ny, nx)
    delta_t  = np.where(mask, idw_t .reshape(ny, nx), 0.0).astype(np.float32)
    delta_td = np.where(mask, idw_td.reshape(ny, nx), 0.0).astype(np.float32)

    n_corrected = int((np.abs(delta_t) > 0.01).sum())
    log.info(f'[mesonet] {n_corrected} gridpoints corrected, '
             f'max|ΔT|={float(np.abs(delta_t).max()):.2f} K  '
             f'max|ΔTd|={float(np.abs(delta_td).max()):.2f} K')
    return delta_t, delta_td
