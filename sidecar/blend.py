import logging
import numpy as np
from scipy.interpolate import RegularGridInterpolator

log = logging.getLogger(__name__)

# Approximate grid spacing for RTMA 2.5km — used for gradient fields (m)
RTMA_DX = 2500.0
RTMA_DY = 2500.0

def _interp_to_rtma(rap_field: np.ndarray,
                    rap_lats: np.ndarray, rap_lons: np.ndarray,
                    rtma_lats: np.ndarray, rtma_lons: np.ndarray) -> np.ndarray:
    """
    Bilinearly interpolate a RAP field (ny_rap, nx_rap) to the RTMA grid.

    rap_lats/lons may be 2D (ny, nx) — only the first col/row is used as the
    1D axis since RAP is on a regular grid.  RegularGridInterpolator requires
    strictly increasing axes, so we flip the lat axis if it is decreasing
    (north-to-south storage).

    Returns float32 array shape (ny_rtma, nx_rtma).
    """
    lats_1d = rap_lats[:, 0] if rap_lats.ndim == 2 else rap_lats
    lons_1d = rap_lons[0, :] if rap_lons.ndim == 2 else rap_lons

    flip_lat = lats_1d[0] > lats_1d[-1]
    field = np.flipud(rap_field) if flip_lat else rap_field
    if flip_lat:
        lats_1d = lats_1d[::-1]

    # fill_value=0 stops linear extrapolation from exploding outside the
    # RAP domain (RTMA extends past RAP edges → NaN/inf without this).
    # Out-of-domain pixels become 0; the iOS draw range filter then hides
    # them automatically (CAPE<100, SRH<25, etc.).
    interp = RegularGridInterpolator(
        (lats_1d, lons_1d), field,
        method='linear', bounds_error=False, fill_value=0.0,
    )
    pts = np.stack([rtma_lats.ravel(), rtma_lons.ravel()], axis=-1)
    out = interp(pts).reshape(rtma_lats.shape)
    return out.astype(np.float32)

def blend(rtma: dict, rap: dict) -> dict:
    """
    Blend RTMA 2.5km surface fields with RAP 13km upper-air fields.

    rtma: output of fetch_rtma  — t2m, td2m, u10, v10, lats, lons
    rap:  output of fetch_rap   — cape, cin, srh1, u500, v500, u10, v10,
                                   t2m_rap, td2m_rap, lats_rap, lons_rap

    Returns dict of float32 grids on the RTMA 2.5km grid, plus lats/lons.
    Missing inputs produce a warning and the dependent params are omitted
    rather than crashing.
    """
    rtma_lats = rtma['lats']    # (1597, 2345)
    rtma_lons = rtma['lons']    # (1597, 2345)
    rap_lats  = rap['lats_rap']
    rap_lons  = rap['lons_rap']

    def interp(field, name):
        """Interpolate one RAP field; return None and warn on failure."""
        if field is None:
            log.warning(f'blend: RAP field "{name}" is None — skipping dependents')
            return None
        try:
            return _interp_to_rtma(field, rap_lats, rap_lons, rtma_lats, rtma_lons)
        except Exception as e:
            log.warning(f'blend: interpolation failed for "{name}": {e}')
            return None

    log.info('Interpolating RAP fields to RTMA 2.5km grid...')
    cape_i   = interp(rap.get('cape'),     'cape')
    cin_i    = interp(rap.get('cin'),      'cin')
    srh1_i   = interp(rap.get('srh1'),     'srh1')
    u500_i   = interp(rap.get('u500'),     'u500')
    v500_i   = interp(rap.get('v500'),     'v500')
    u10_rap  = interp(rap.get('u10'),      'u10_rap')
    v10_rap  = interp(rap.get('v10'),      'v10_rap')
    t2m_rap  = interp(rap.get('t2m_rap'),  't2m_rap')
    td2m_rap = interp(rap.get('td2m_rap'), 'td2m_rap')
    log.info('Interpolation complete. Deriving blended parameters...')

    # RTMA surface fields — already on 2.5km grid
    t2m  = rtma['t2m']     # K
    td2m = rtma['td2m']    # K
    u10  = rtma['u10']     # m/s
    v10  = rtma['v10']     # m/s

    out: dict = {}

    # --- SBCAPE: RAP CAPE corrected by RTMA vs RAP surface-T delta -------
    # RTMA has higher-resolution, assimilated surface T/Td. A 1 K warmer
    # surface shifts SBCAPE by ~180 J/kg (rule of thumb from operational NWP).
    if cape_i is not None and t2m_rap is not None:
        delta_t = t2m - t2m_rap
        out['sbcape'] = np.maximum(0, cape_i + delta_t * 180.0).astype(np.float32)
    elif cape_i is not None:
        out['sbcape'] = np.maximum(0, cape_i)
        log.warning('blend: sbcape has no surface-T correction (t2m_rap missing)')
    else:
        log.warning('blend: sbcape skipped (cape missing)')

    # --- SBCIN: interpolated RAP CIN ------------------------------------
    if cin_i is not None:
        out['sbcin'] = cin_i
    else:
        log.warning('blend: sbcin skipped (cin missing)')

    # --- 0-1km SRH: interpolated directly from RAP ----------------------
    if srh1_i is not None:
        out['srh1'] = srh1_i
    else:
        log.warning('blend: srh1 skipped (hlcy missing)')

    # --- 0-6km BWD: RTMA 10m surface + RAP 500mb upper ------------------
    # True 0-6km layer winds aren't in awp130p; 500mb (~5500m AGL) is the
    # best available proxy.
    if u500_i is not None and v500_i is not None:
        out['bwd6'] = np.sqrt((u500_i - u10)**2 + (v500_i - v10)**2).astype(np.float32)
    else:
        log.warning('blend: bwd6 skipped (u500/v500 missing)')

    # --- LCL height: Bolton (1980) using RTMA T/Td ----------------------
    lcl = 125.0 * (t2m - td2m)    # meters AGL, float64 intermediate OK

    # --- Fixed-layer STP (Thompson et al. 2003) -------------------------
    if 'sbcape' in out and 'srh1' in out and 'bwd6' in out:
        cape_term  = out['sbcape'] / 1500.0
        lcl_term   = np.clip((2000.0 - lcl) / 1000.0, 0.0, 1.0)
        srh_term   = out['srh1'] / 150.0
        shear_term = np.minimum(1.5, out['bwd6'] / 10.288)
        out['stp'] = np.maximum(0, cape_term * lcl_term * srh_term * shear_term).astype(np.float32)
    else:
        log.warning('blend: stp skipped (sbcape, srh1, or bwd6 missing)')

    # --- Surface relative vorticity: dv/dx - du/dy (s⁻¹) ----------------
    # Cyclonic (counterclockwise) vorticity is positive in the N. hemisphere.
    dvdx = np.gradient(v10, axis=1) / RTMA_DX
    dudy = np.gradient(u10, axis=0) / RTMA_DY
    out['vort'] = (dvdx - dudy).astype(np.float32)

    # --- Surface convergence: -(du/dx + dv/dy) ---------------------------
    # Positive = convergent (inflow).
    dudx = np.gradient(u10, axis=1) / RTMA_DX
    dvdy = np.gradient(v10, axis=0) / RTMA_DY
    out['conv'] = (-(dudx + dvdy)).astype(np.float32)

    # --- Td depression: T - Td (K) ---------------------------------------
    # Lower values = more moist; useful as a dryline proxy.
    out['td_dep'] = (t2m - td2m).astype(np.float32)

    # Pass grid coordinates through for writer.py bbox calculation
    out['lats'] = rtma_lats
    out['lons'] = rtma_lons

    ready = [k for k in out if k not in ('lats', 'lons')]
    log.info(f'Blend complete: {ready}')
    return out
