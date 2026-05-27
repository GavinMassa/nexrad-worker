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
    # Use the centre column/row for the 1D axis rather than column 0 / row 0.
    # On a Lambert Conformal grid the outer column is curved — column 0 lats
    # are not monotonic, which causes RegularGridInterpolator to project RAP
    # fields onto wrong positions (~2-3° geographic shift in STP contours).
    # The centre column is closest to the projection's standard meridian and
    # is monotonic by construction for any standard CONUS RAP domain.
    if rap_lats.ndim == 2:
        mid_col = rap_lats.shape[1] // 2
        lats_1d = rap_lats[:, mid_col]
    else:
        lats_1d = rap_lats

    if rap_lons.ndim == 2:
        mid_row = rap_lons.shape[0] // 2
        lons_1d = rap_lons[mid_row, :]
    else:
        lons_1d = rap_lons

    log.info(f'[blend] RAP lats_1d: min={lats_1d.min():.2f} max={lats_1d.max():.2f} '
             f'monotonic={bool(np.all(np.diff(lats_1d) > 0) or np.all(np.diff(lats_1d) < 0))}')
    log.info(f'[blend] RAP lons_1d: min={lons_1d.min():.2f} max={lons_1d.max():.2f} '
             f'monotonic={bool(np.all(np.diff(lons_1d) > 0) or np.all(np.diff(lons_1d) < 0))}')
    log.info(f'[blend] RTMA lats: min={rtma_lats.min():.2f} max={rtma_lats.max():.2f}')
    log.info(f'[blend] RTMA lons: min={rtma_lons.min():.2f} max={rtma_lons.max():.2f}')

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


def apply_tpw_correction(
    td2m:      np.ndarray,   # RTMA Td (ny, nx) K
    rtma_lats: np.ndarray,   # (ny, nx)
    rtma_lons: np.ndarray,   # (ny, nx) 0-360 or -180..180
    tpw_data:  dict | None,  # output of fetch_tpw._extract_tpw or None
) -> np.ndarray:
    """
    Blend GOES-18 TPW into RTMA Td to correct moisture depth.

    Method (Spar & Thompson 2019 approximation):
      1. Convert RTMA Td to equivalent TPW using Bolton (1980) approximation:
           e_s = 6.112 * exp(17.67 * (Td-273.15) / (Td-273.15+243.5))  hPa
           TPW_rtma ≈ 2.0 * e_s  (rule of thumb, ~valid for 1000-850mb layer)
      2. Interpolate GOES TPW to RTMA grid using KDTree nearest-neighbor
      3. Compute TPW innovation: delta_tpw = TPW_goes - TPW_rtma
      4. Convert TPW innovation back to Td correction via inverse Bolton:
           delta_Td ≈ delta_tpw / (d(TPW)/d(Td))
      5. Clamp correction to ±3K, apply to RTMA Td

    Returns corrected Td (float32, same shape as input).
    Leaves Td unchanged if tpw_data is None.
    """
    if tpw_data is None:
        return td2m

    try:
        from scipy.spatial import KDTree

        tpw_arr  = tpw_data['tpw']    # (ny_goes, nx_goes) mm, NaN over no-data
        tpw_lats = tpw_data['lats']
        tpw_lons = tpw_data['lons']

        # Normalise GOES lons to same convention as RTMA lons
        rtma_lons_norm = np.where(rtma_lons > 180.0, rtma_lons - 360.0, rtma_lons)
        tpw_lons_norm  = np.where(tpw_lons  > 180.0, tpw_lons  - 360.0, tpw_lons)

        # Build KDTree over valid GOES TPW points only
        valid_mask = np.isfinite(tpw_arr)
        if valid_mask.sum() < 1000:
            log.warning('[tpw] too few valid TPW pixels — skipping correction')
            return td2m

        valid_lats = tpw_lats[valid_mask]
        valid_lons = tpw_lons_norm[valid_mask]
        valid_tpw  = tpw_arr[valid_mask]

        tpw_pts = np.column_stack([valid_lats, valid_lons])
        tree    = KDTree(tpw_pts)

        # Interpolate to RTMA grid
        rtma_pts = np.column_stack([rtma_lats.ravel(), rtma_lons_norm.ravel()])
        dists, idx = tree.query(rtma_pts, k=1)

        # Mask out RTMA points too far from any GOES pixel (>0.1° ≈ 11km)
        far_mask = dists > 0.1
        tpw_interp = valid_tpw[idx].reshape(rtma_lats.shape)
        tpw_interp[far_mask.reshape(rtma_lats.shape)] = np.nan

        # RTMA Td → equivalent TPW (Bolton approximation)
        td_c     = td2m - 273.15   # K → °C
        e_s      = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))  # hPa
        tpw_rtma = 2.0 * e_s       # mm (approximate)

        # TPW innovation → Td correction
        delta_tpw = tpw_interp - tpw_rtma   # mm
        # Inverse Bolton: dTd ≈ delta_tpw / (d(TPW)/d(Td))
        # d(TPW)/d(Td) = 2 * d(e_s)/d(Td) = 2 * e_s * 17.67*243.5 / (td_c+243.5)^2
        dtpw_dtd = 2.0 * e_s * 17.67 * 243.5 / (td_c + 243.5)**2
        dtpw_dtd = np.maximum(dtpw_dtd, 0.1)   # avoid division by zero
        delta_td = delta_tpw / dtpw_dtd         # K

        # Clamp to ±3K — prevents artifacts from GOES retrieval errors
        delta_td = np.clip(delta_td, -3.0, 3.0)
        delta_td = np.where(np.isfinite(delta_td), delta_td, 0.0)

        n_corrected = int((np.abs(delta_td) > 0.1).sum())
        log.info(f'[tpw] {n_corrected} gridpoints corrected, '
                 f'max|ΔTd|={float(np.abs(delta_td).max()):.2f}K')

        return (td2m + delta_td).astype(np.float32)

    except Exception as e:
        log.warning(f'[tpw] correction failed: {e} — using raw RTMA Td')
        return td2m


def blend(rtma: dict, rap: dict, tpw_data: dict | None = None) -> dict:
    """
    Blend RTMA 2.5km surface fields with RAP 13km upper-air fields.

    rtma:     output of fetch_rtma  — t2m, td2m, u10, v10, lats, lons
    rap:      output of fetch_rap   — cape, cin, srh1, u500, v500, u850, v850,
                                      u10, v10, t2m_rap, td2m_rap, lats_rap, lons_rap
    tpw_data: output of fetch_tpw   — tpw, lats, lons (float32 2D arrays); or None

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
    u850_i   = interp(rap.get('u850'),     'u850')
    v850_i   = interp(rap.get('v850'),     'v850')
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

    # Apply GOES-18 TPW correction to RTMA Td BEFORE parcel calculations.
    # This corrects for moisture depth (column moisture) not captured by
    # surface 2m Td alone. Applied first so LCL, SBCAPE correction, and
    # Td depression all benefit from the improved Td.
    if tpw_data is not None:
        log.info('[tpw] applying GOES-18 TPW moisture correction...')
        td2m = apply_tpw_correction(td2m, rtma_lats, rtma_lons, tpw_data)
    else:
        log.info('[tpw] no TPW data — using raw RTMA Td')

    out: dict = {}

    # --- SBCAPE: RAP CAPE corrected by RTMA vs RAP surface-T delta -------
    # RTMA has higher-resolution, assimilated surface T/Td. A 1 K warmer
    # surface shifts SBCAPE by ~180 J/kg (rule of thumb from operational NWP).
    # Clamp delta_t to ±2 K to prevent spurious correction from interpolation
    # artifacts (e.g. RAP extrapolation over Great Lakes → unclamped delta can
    # reach +15 K → +2700 J/kg phantom CAPE over open water at night).
    # Also suppress the correction entirely where RAP CAPE is near-zero
    # (stable/water areas) — no need to shift 0 J/kg by surface-T bias.
    if cape_i is not None and t2m_rap is not None:
        delta_t_clamped  = np.clip(t2m - t2m_rap, -2.0, 2.0)
        sbcape_corrected = np.maximum(0, cape_i + delta_t_clamped * 180.0)
        cape_mask        = cape_i < 50.0          # stable / water / no CAPE
        out['sbcape']    = np.where(
            cape_mask,
            np.maximum(0, cape_i),                # keep raw RAP value (≈ 0)
            sbcape_corrected,
        ).astype(np.float32)
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

    # --- 0-6km BWD: two-layer approximation using 850mb + 500mb + 10m --------
    # Layer 1: 10m → 850mb (~1500m AGL) captures low-level hodograph curvature
    # Layer 2: 850mb → 500mb (~1500m → 5500m AGL) captures mid-level shear
    # Sum of vector magnitudes approximates total 0-6km BWD, handling the
    # curvature common in LJ environments and QLCS setups better than a
    # single 500mb-10m difference.
    # Falls back to single-layer 500mb-10m if 850mb is missing.
    # Gate: values below 30 m/s are zeroed — iOS renderer skips 0-value cells,
    # so only operationally significant shear (≥30 m/s) produces contours.
    BWD6_MIN = 15.0   # m/s — display gate
    if u500_i is not None and v500_i is not None:
        if u850_i is not None and v850_i is not None:
            # Two-layer: |V850-V10| + |V500-V850|
            layer1 = np.sqrt((u850_i - u10)**2 + (v850_i - v10)**2)
            layer2 = np.sqrt((u500_i - u850_i)**2 + (v500_i - v850_i)**2)
            raw = (layer1 + layer2).astype(np.float32)
            log.info('bwd6: two-layer (10m→850mb→500mb)')
        else:
            # Fallback: single-layer 500mb-10m
            raw = np.sqrt((u500_i - u10)**2 + (v500_i - v10)**2).astype(np.float32)
            log.warning('bwd6: fallback to single-layer (u850/v850 missing)')
        out['bwd6'] = np.where(raw >= BWD6_MIN, raw, 0.0).astype(np.float32)
        log.info(f'bwd6: gate={BWD6_MIN}m/s, active={int((raw >= BWD6_MIN).sum())} cells')
    else:
        log.warning('blend: bwd6 skipped (u500/v500 missing)')

    # --- LCL height: Bolton (1980) using RTMA T/Td ----------------------
    # Uses TPW-corrected td2m if tpw_data was provided above.
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
    # Uses TPW-corrected td2m if tpw_data was provided above.
    out['td_dep'] = (t2m - td2m).astype(np.float32)

    # Pass grid coordinates through for writer.py bbox calculation
    out['lats'] = rtma_lats
    out['lons'] = rtma_lons

    ready = [k for k in out if k not in ('lats', 'lons')]
    log.info(f'Blend complete: {ready}')
    return out
