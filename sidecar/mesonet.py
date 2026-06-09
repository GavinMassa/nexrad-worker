import logging
import numpy as np
import httpx
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# IEM ASOS/mesonet current obs endpoint.
# Returns all US stations with obs within the last 90 minutes.
IEM_CURRENT_URL = 'https://mesonet.agron.iastate.edu/api/1/currents.geojson'


async def fetch_mesonet_obs() -> list[dict] | None:
    """
    Fetch current surface obs from IEM.
    Returns list of dicts with keys: lat, lon, t_k, td_k (Kelvin),
    psfc_mb (station pressure in mb, or None if not reported).
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
            # Altimeter setting → station pressure (mb).
            # We lack per-station elevation from IEM, so we approximate station
            # pressure by the altimeter setting itself (alti ≈ P_stn within ~5 mb
            # for stations below ~500 m, covering >95% of the ASOS network).
            # alti is inches Hg; 1 inHg = 33.8639 mb. Parse defensively so a bad
            # pressure value never discards an otherwise-good T/Td observation.
            alti = props.get('alti')
            try:
                psfc_mb = float(alti) * 33.8639 if alti is not None else None
            except (TypeError, ValueError):
                psfc_mb = None
            # Sanity check: valid surface pressure 850-1050 mb
            if psfc_mb is not None and not (850.0 < psfc_mb < 1050.0):
                psfc_mb = None
            stations.append({
                'lat':     float(coords[1]),
                'lon':     float(coords[0]),
                't_k':     (float(tmpf) - 32.0) * 5.0 / 9.0 + 273.15,
                'td_k':    (float(dwpf) - 32.0) * 5.0 / 9.0 + 273.15,
                'psfc_mb': psfc_mb,   # None if not reported / out of range
            })
        except (TypeError, ValueError):
            continue

    log.info(f'[mesonet] {len(stations)} stations with valid T/Td')
    return stations if stations else None


def compute_correction(
    stations:   list[dict],
    grid_lats:  np.ndarray,   # (ny, nx) downsampled RTMA lats
    grid_lons:  np.ndarray,   # (ny, nx) downsampled RTMA lons, -180..180
    grid_t2m:   np.ndarray,   # (ny, nx) K background temperature
    grid_td2m:  np.ndarray,   # (ny, nx) K background dewpoint
    grid_psfc:  np.ndarray,   # (ny, nx) mb background surface pressure
                              #   (use 1013.25 * np.ones_like(grid_t2m) if unavailable)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2-pass Barnes OA using KDTree radius queries — no full distance matrix.

    Memory usage: O(n_gp × k) where k = avg stations per radius (~5-20),
    vs O(n_gp × n_st) for the old full-matrix IDW. Safe at any station count.

    Barnes parameters (matching SPC SFCOA scale):
      κ  = 0.3  (deg²) — first-pass influence radius ~0.55° ≈ 60 km
      γ  = 0.3  — second-pass convergence factor
      R  = 2.0° — hard cutoff radius for KDTree query (~220 km)

    Returns (delta_t, delta_td, delta_psfc): float32 arrays shape (ny, nx).
    delta_psfc is 0 where no pressure obs are available.
    """
    from scipy.spatial import KDTree

    ny, nx = grid_t2m.shape
    n_gp   = ny * nx

    if not stations:
        z = np.zeros((ny, nx), np.float32)
        return z, z, z

    # ── Station arrays ────────────────────────────────────────────────────────
    st_lats   = np.array([s['lat']  for s in stations], dtype=np.float64)
    st_lons   = np.array([s['lon']  for s in stations], dtype=np.float64)
    st_t      = np.array([s['t_k']  for s in stations], dtype=np.float32)
    st_td     = np.array([s['td_k'] for s in stations], dtype=np.float32)
    st_p_raw  = np.array([s.get('psfc_mb') if s.get('psfc_mb') is not None else np.nan
                          for s in stations], dtype=np.float32)
    has_p     = ~np.isnan(st_p_raw)

    # ── Background at each station via KDTree ─────────────────────────────────
    gp_lats = grid_lats.ravel().astype(np.float64)
    gp_lons = grid_lons.ravel().astype(np.float64)
    gp_pts  = np.column_stack([gp_lats, gp_lons])
    tree_gp = KDTree(gp_pts)

    st_pts     = np.column_stack([st_lats, st_lons])
    _, st_idx  = tree_gp.query(st_pts)
    iy, ix     = np.unravel_index(st_idx, (ny, nx))

    bg_t   = grid_t2m [iy, ix].astype(np.float32)
    bg_td  = grid_td2m[iy, ix].astype(np.float32)
    bg_p   = grid_psfc[iy, ix].astype(np.float32)

    innov_t  = st_t  - bg_t
    innov_td = st_td - bg_td
    innov_p  = np.where(has_p, st_p_raw - bg_p, np.nan)

    # QC: discard stations with implausible innovations
    MAX_T_INNOV  = 8.0   # K
    MAX_P_INNOV  = 15.0  # mb
    qc_t  = np.abs(innov_t)  < MAX_T_INNOV
    qc_td = np.abs(innov_td) < MAX_T_INNOV
    qc_p  = has_p & (np.abs(innov_p) < MAX_P_INNOV)
    qc_ok = qc_t & qc_td   # T and Td must both pass for a station to contribute
    n_bad = int((~qc_ok).sum())
    if n_bad:
        log.info(f'[mesonet] QC removed {n_bad} stations')

    # ── Barnes parameters ─────────────────────────────────────────────────────
    KAPPA  = 0.3    # deg² — first-pass Gaussian width
    GAMMA  = 0.3    # second-pass convergence factor
    KAPPA2 = KAPPA * GAMMA
    R      = 2.0    # deg — hard cutoff radius

    # ── KDTree over station positions ─────────────────────────────────────────
    tree_st = KDTree(st_pts)

    # ── Initialise output arrays ──────────────────────────────────────────────
    ana1_t  = np.zeros(n_gp, dtype=np.float32)   # first-pass analysis increment
    ana1_td = np.zeros(n_gp, dtype=np.float32)
    ana1_p  = np.zeros(n_gp, dtype=np.float32)
    cnt1    = np.zeros(n_gp, dtype=np.int32)

    # Process in chunks to keep peak memory bounded (~50 MB per chunk of 10k pts)
    CHUNK = 10_000
    for start in range(0, n_gp, CHUNK):
        end    = min(start + CHUNK, n_gp)
        q_pts  = gp_pts[start:end]               # (chunk, 2)
        # Radius query: returns list of lists (variable length per point)
        nbrs   = tree_st.query_ball_point(q_pts, r=R)

        for local_i, nn_idx in enumerate(nbrs):
            gi = start + local_i
            if not nn_idx:
                continue
            nn_idx = np.array(nn_idx, dtype=np.int32)
            valid  = nn_idx[qc_ok[nn_idx]]
            if len(valid) < 3:      # need ≥3 stations for reliable analysis
                continue

            dlat = gp_lats[gi] - st_lats[valid]
            dlon = gp_lons[gi] - st_lons[valid]
            d2   = dlat**2 + dlon**2

            w    = np.exp(-d2 / KAPPA)
            ws   = w.sum()
            if ws < 1e-10:
                continue

            ana1_t [gi] = float((w * innov_t [valid]).sum() / ws)
            ana1_td[gi] = float((w * innov_td[valid]).sum() / ws)
            cnt1   [gi] = len(valid)

            # Pressure: only where obs available and QC passed
            p_valid = valid[qc_p[valid]]
            if len(p_valid) >= 2:
                wp  = np.exp(-((gp_lats[gi] - st_lats[p_valid])**2 +
                               (gp_lons[gi] - st_lons[p_valid])**2) / KAPPA)
                wps = wp.sum()
                if wps > 1e-10:
                    ana1_p[gi] = float((wp * innov_p[p_valid]).sum() / wps)

    # ── Second pass: correct residuals ───────────────────────────────────────
    # Compute first-pass estimate at each station, then interpolate residual
    ana2_t  = np.zeros(n_gp, dtype=np.float32)
    ana2_td = np.zeros(n_gp, dtype=np.float32)

    for start in range(0, n_gp, CHUNK):
        end   = min(start + CHUNK, n_gp)
        q_pts = gp_pts[start:end]
        nbrs  = tree_st.query_ball_point(q_pts, r=R)

        for local_i, nn_idx in enumerate(nbrs):
            gi = start + local_i
            if cnt1[gi] < 3:
                continue
            nn_idx = np.array(nn_idx, dtype=np.int32)
            valid  = nn_idx[qc_ok[nn_idx]]
            if len(valid) < 3:
                continue

            dlat = gp_lats[gi] - st_lats[valid]
            dlon = gp_lons[gi] - st_lons[valid]
            d2   = dlat**2 + dlon**2

            # First-pass estimate at each station = ana1 at the station's nearest
            # gridpoint. Residual is obs-innovation minus that first-pass estimate.
            st_gp_idx   = st_idx[valid]
            fp_at_st_t  = ana1_t [st_gp_idx]
            fp_at_st_td = ana1_td[st_gp_idx]

            resid_t  = innov_t [valid] - fp_at_st_t
            resid_td = innov_td[valid] - fp_at_st_td

            w2  = np.exp(-d2 / KAPPA2)
            ws2 = w2.sum()
            if ws2 < 1e-10:
                continue

            ana2_t [gi] = float((w2 * resid_t ).sum() / ws2)
            ana2_td[gi] = float((w2 * resid_td).sum() / ws2)

    # ── Combine passes and reshape ────────────────────────────────────────────
    mask = (cnt1 >= 3).reshape(ny, nx)

    delta_t  = np.where(mask, (ana1_t  + ana2_t ).reshape(ny, nx), 0.0)
    delta_td = np.where(mask, (ana1_td + ana2_td).reshape(ny, nx), 0.0)
    delta_p  = np.where(mask, ana1_p   .reshape(ny, nx), 0.0)   # single-pass for P

    # Clamp to physical limits
    delta_t  = np.clip(delta_t,  -12.0, 15.0).astype(np.float32)
    delta_td = np.clip(delta_td, -12.0, 15.0).astype(np.float32)
    delta_p  = np.clip(delta_p,  -20.0, 20.0).astype(np.float32)

    n_corrected = int((np.abs(delta_t) > 0.05).sum())
    log.info(f'[mesonet] correction written: {len(stations)} stations, '
             f'{ny}×{nx} grid')
    log.info(f'[mesonet] {n_corrected} gridpoints corrected, '
             f'max|ΔT|={float(np.abs(delta_t).max()):.2f} K  '
             f'max|ΔTd|={float(np.abs(delta_td).max()):.2f} K  '
             f'max|ΔP|={float(np.abs(delta_p).max()):.2f} mb')
    return delta_t, delta_td, delta_p
