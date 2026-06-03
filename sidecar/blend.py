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


def compute_baci(
    rtma:      dict,
    rap:       dict,
    rap_lats:  np.ndarray,
    rap_lons:  np.ndarray,
    rtma_lats: np.ndarray,
    rtma_lons: np.ndarray,
) -> np.ndarray:
    """
    Bay Area Convection Index — terrain-aware convection initiation risk.
    Valid for lat 36.5–38.5, lon -123 to -120 (Santa Cruz Mtns, Diablo Range,
    Central Valley). Zero outside this domain.
    """
    from pathlib import Path

    TERRAIN_PATH  = Path('/app/sidecar-out/baci_terrain.npz')
    BACI_LAT_MIN, BACI_LAT_MAX =  36.5,  38.5
    BACI_LON_MIN, BACI_LON_MAX = -123.0, -120.0

    ny_rtma, nx_rtma = rtma_lats.shape
    baci_out = np.zeros((ny_rtma, nx_rtma), dtype=np.float32)

    lons_180    = np.where(rtma_lons > 180.0, rtma_lons - 360.0, rtma_lons)
    domain_mask = (
        (rtma_lats >= BACI_LAT_MIN) & (rtma_lats <= BACI_LAT_MAX) &
        (lons_180  >= BACI_LON_MIN) & (lons_180  <= BACI_LON_MAX)
    )
    if domain_mask.sum() < 100:
        log.warning('[BACI] domain mask empty — RTMA may not cover Bay Area')
        return baci_out

    def interp_rap(field, name):
        if field is None:
            log.warning(f'[BACI] RAP field {name} missing')
            return None
        try:
            return _interp_to_rtma(field, rap_lats, rap_lons, rtma_lats, rtma_lons)
        except Exception as e:
            log.warning(f'[BACI] interp failed for {name}: {e}')
            return None

    cape_i  = interp_rap(rap.get('cape'),  'cape')
    cin_i   = interp_rap(rap.get('cin'),   'cin')
    t925_i  = interp_rap(rap.get('t925'),  't925')
    t700_i  = interp_rap(rap.get('t700'),  't700')
    td700_i = interp_rap(rap.get('td700'), 'td700')
    pwat_i  = interp_rap(rap.get('pwat'),  'pwat')
    u850_i  = interp_rap(rap.get('u850'),  'u850')
    v850_i  = interp_rap(rap.get('v850'),  'v850')

    t2m = rtma['t2m']

    # Term 1: CAPE — 1.0 at 500 J/kg (validated: 568 J/kg on 2026-05-27)
    if cape_i is None:
        log.warning('[BACI] skipping — cape missing')
        return baci_out
    cape_term = np.clip(cape_i / 500.0, 0.0, 1.0)

    # Term 2: No-cap — validated: CIN=0 on event day → nocin_term=1.0
    if cin_i is None:
        nocin_term = np.ones_like(cape_term)
    else:
        nocin_term = np.clip(1.0 - np.abs(cin_i) / 50.0, 0.0, 1.0)

    # Term 3: 0-3km lapse rate — validated: 9.3°C/km on event day
    if t925_i is None:
        lapse_term = np.ones_like(cape_term) * 0.5
    else:
        lr_0_3km   = (t2m - t925_i) / 0.750   # °C/km
        lapse_term = np.clip((lr_0_3km - 7.0) / 3.0, 0.0, 1.0)

    # Term 4: PWAT — validated: 18.3mm on event day → pwat_term=0.64
    if pwat_i is None:
        pwat_term = np.ones_like(cape_term) * 0.5
    else:
        pwat_mm   = np.where(pwat_i < 3.0, pwat_i * 25.4, pwat_i)
        pwat_term = np.clip((pwat_mm - 10.0) / 15.0, 0.0, 1.0)

    # Term 5: 700mb RH — validated: 79% on event day → rh_term=0.73
    if t700_i is None or td700_i is None:
        rh_term = np.ones_like(cape_term) * 0.5
    else:
        td700_c = np.clip(td700_i - 273.15, -60.0, 30.0)
        t700_c  = np.clip(t700_i  - 273.15, -60.0, 30.0)
        rh_700  = 100.0 * np.exp(17.67 * td700_c / (td700_c + 243.5)) / \
                          np.exp(17.67 * t700_c  / (t700_c  + 243.5))
        rh_700  = np.clip(rh_700, 0.0, 100.0)
        rh_term = np.clip((rh_700 - 50.0) / 40.0, 0.0, 1.0)

    # Term 6: upslope 850mb wind component perpendicular to terrain
    upslope_term = np.ones_like(cape_term) * 0.5

    if TERRAIN_PATH.exists() and u850_i is not None and v850_i is not None:
        try:
            from scipy.spatial import KDTree
            terrain    = np.load(str(TERRAIN_PATH))
            aspect_deg = terrain['aspect']
            t_lats     = terrain['lats']
            t_lons     = terrain['lons']

            t_lons_2d, t_lats_2d = np.meshgrid(t_lons, t_lats)
            tree = KDTree(np.column_stack([t_lats_2d.ravel(), t_lons_2d.ravel()]))

            rtma_lons_180 = np.where(rtma_lons > 180.0, rtma_lons - 360.0, rtma_lons)
            _, idx        = tree.query(
                np.column_stack([rtma_lats.ravel(), rtma_lons_180.ravel()]), k=1
            )
            aspect_interp = aspect_deg.ravel()[idx].reshape(ny_rtma, nx_rtma)

            uphill_rad   = np.radians(aspect_interp + 180.0)
            upslope_ms   = u850_i * np.sin(uphill_rad) + v850_i * np.cos(uphill_rad)
            upslope_term = np.clip(upslope_ms / 8.0, 0.0, 1.0)
            log.info(f'[BACI] upslope: max={float(upslope_term.max()):.2f}, '
                     f'domain mean={float(upslope_term[domain_mask].mean()):.2f}')
        except Exception as e:
            log.warning(f'[BACI] terrain aspect failed: {e} — using default')

    baci_full = (cape_term * nocin_term * lapse_term *
                 pwat_term * rh_term * upslope_term).astype(np.float32)
    baci_out  = np.where(domain_mask, baci_full, 0.0).astype(np.float32)

    log.info(f'[BACI] n≥0.25={int((baci_out >= 0.25).sum())}, '
             f'n≥0.5={int((baci_out >= 0.5).sum())}, '
             f'max={float(baci_out.max()):.3f}')
    return baci_out


def blend(rtma: dict, rap: dict, tpw_data: dict | None = None,
          hrrr_hlcy: dict | None = None) -> dict:
    """
    Blend RTMA 2.5km surface fields with RAP 13km upper-air fields.

    rtma:      output of fetch_rtma   — t2m, td2m, u10, v10, lats, lons
    rap:       output of fetch_rap    — cape, cin, srh1, u500, v500, u850, v850,
                                        u10, v10, t2m_rap, td2m_rap, lats_rap, lons_rap
    tpw_data:  output of fetch_tpw    — tpw, lats, lons (float32 2D arrays); or None
    hrrr_hlcy: output of fetch_hrrr_hlcy — hlcy, lats, lons (float32 2D arrays); or None
               HRRR 0-1km SRH at native 3km resolution; used as second backstop
               in the SRH blend. Failure to provide this does not abort the blend.

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

    _mid_col = rap_lats.shape[1] // 2 if rap_lats.ndim == 2 else 0
    _mid_row = rap_lons.shape[0] // 2 if rap_lons.ndim == 2 else 0
    _lats_1d = rap_lats[:, _mid_col] if rap_lats.ndim == 2 else rap_lats
    _lons_1d = rap_lons[_mid_row, :] if rap_lons.ndim == 2 else rap_lons
    log.info(f'[blend] RAP lats_1d: min={_lats_1d.min():.2f} max={_lats_1d.max():.2f} '
             f'monotonic={bool(np.all(np.diff(_lats_1d) > 0) or np.all(np.diff(_lats_1d) < 0))}')
    log.info(f'[blend] RAP lons_1d: min={_lons_1d.min():.2f} max={_lons_1d.max():.2f} '
             f'monotonic={bool(np.all(np.diff(_lons_1d) > 0) or np.all(np.diff(_lons_1d) < 0))}')
    log.info(f'[blend] RTMA lats: min={rtma_lats.min():.2f} max={rtma_lats.max():.2f}')
    log.info(f'[blend] RTMA lons: min={rtma_lons.min():.2f} max={rtma_lons.max():.2f}')
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
    u925_i   = interp(rap.get('u925'),     'u925')
    v925_i   = interp(rap.get('v925'),     'v925')
    u950_i   = interp(rap.get('u950'),     'u950')
    v950_i   = interp(rap.get('v950'),     'v950')
    ustm_i   = interp(rap.get('ustm'),     'ustm')
    vstm_i   = interp(rap.get('vstm'),     'vstm')
    t2m_rap  = interp(rap.get('t2m_rap'),  't2m_rap')
    td2m_rap = interp(rap.get('td2m_rap'), 'td2m_rap')
    cape3k_i = interp(rap.get('cape3k'),   'cape3k')
    mucape_i = interp(rap.get('mucape'),   'mucape')
    t700_i   = interp(rap.get('t700'),     't700')

    # Interpolate HRRR 0-1km HLCY to RTMA grid (optional — 3km → 2.5km).
    # HRRR uses Lambert Conformal like RAP but at 3km resolution; the same
    # _interp_to_rtma() bilinear interpolator handles it without modification.
    # HRRR lons are native -180..180 so no normalisation is needed.
    hrrr_srh_i = None
    if hrrr_hlcy is not None:
        try:
            hrrr_srh_i = _interp_to_rtma(
                hrrr_hlcy['hlcy'],
                hrrr_hlcy['lats'],
                hrrr_hlcy['lons'],
                rtma_lats,
                rtma_lons,
            )
            log.info(f'[hrrr] hlcy interpolated to RTMA grid: '
                     f'max={float(hrrr_srh_i.max()):.0f} '
                     f'shape={hrrr_srh_i.shape}')
        except Exception as e:
            log.warning(f'[hrrr] hlcy interpolation failed: {e}')
            hrrr_srh_i = None

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

    # Shared surface-T delta — used by both SBCAPE and SBCIN corrections.
    # Computed here so both blocks can reference it without duplication.
    if t2m_rap is not None:
        delta_t_clamped = np.clip(t2m - t2m_rap, -2.0, 2.0)
    else:
        delta_t_clamped = np.zeros_like(t2m)

    # --- SBCAPE: RAP CAPE corrected by RTMA vs RAP surface-T delta -------
    SBCAPE_MIN = 150.0   # J/kg — display gate
    if cape_i is not None and t2m_rap is not None:
        sbcape_corrected = np.maximum(0, cape_i + delta_t_clamped * 180.0)
        cape_mask        = cape_i < 50.0
        raw_sbcape       = np.where(cape_mask, np.maximum(0, cape_i), sbcape_corrected).astype(np.float32)
        out['sbcape']    = np.where(raw_sbcape >= SBCAPE_MIN, raw_sbcape, 0.0).astype(np.float32)
    elif cape_i is not None:
        raw_sbcape    = np.maximum(0, cape_i).astype(np.float32)
        out['sbcape'] = np.where(raw_sbcape >= SBCAPE_MIN, raw_sbcape, 0.0).astype(np.float32)
        log.warning('blend: sbcape has no surface-T correction (t2m_rap missing)')
    else:
        log.warning('blend: sbcape skipped (cape missing)')

    # --- SBCIN: RAP CIN with RTMA linear thermal correction -------------
    # Linear additive correction: 1K surface warming reduces cap by 40 J/kg,
    # 1K cooling amplifies it. 40 J/kg/K is the empirical mixed-layer sensitivity.
    # Uses additive (not multiplicative) approach because CIN is a negative
    # energy quantity — exponential multipliers on negative numbers produce
    # unbounded amplification when delta_t is negative (cold air advection).
    # Clamp to [-300, 0] — operationally, CIN below −300 J/kg is "fully capped"
    # regardless of exact value (no surface parcel breaks it). Clamping at −300
    # prevents stable non-convective regimes (Rockies, polar air) from producing
    # extreme negative values that drag the CIN gate down to zero everywhere.
    if cin_i is not None:
        if t2m_rap is not None:
            cin_correction = delta_t_clamped * 40.0
            out['sbcin'] = np.clip(
                cin_i + cin_correction, -300.0, 0.0
            ).astype(np.float32)
            log.info(f'[sbcin] linear thermal correction: '
                     f'min={float(out["sbcin"].min()):.0f} '
                     f'correction range=[{float(cin_correction.min()):.0f}, '
                     f'{float(cin_correction.max()):.0f}] J/kg')
        else:
            out['sbcin'] = np.clip(cin_i, -300.0, 0.0).astype(np.float32)
            log.warning('blend: sbcin has no thermal correction (t2m_rap missing)')
    else:
        log.warning('blend: sbcin skipped (cin missing)')

    # --- 0-3km CAPE: low-level buoyancy from RAP lev_0-3000_m layer ------
    # No RTMA correction applied — the layer-mean CAPE is not sensitive to the
    # surface-skin T anomaly the way surface-based CAPE is. Pass through with
    # floor clamp and display gate only.
    CAPE3K_MIN = 50.0   # J/kg — display gate (remove noise)
    if cape3k_i is not None:
        out['cape3k'] = np.where(
            np.maximum(0.0, cape3k_i) >= CAPE3K_MIN,
            np.maximum(0.0, cape3k_i), 0.0,
        ).astype(np.float32)
        log.info(f'[cape3k] max={float(out["cape3k"].max()):.0f} J/kg '
                 f'active={int((out["cape3k"] > 0).sum())} cells')
    else:
        log.warning('blend: cape3k skipped (RAP 0-3km CAPE not available)')

    # --- MUCAPE: most unstable parcel CAPE (180-0mb search layer) -----------
    # No RTMA surface correction — MU parcel is elevated, not surface-based.
    # Same display gate as SBCAPE (150 J/kg) to suppress noise.
    MUCAPE_MIN = 150.0
    if mucape_i is not None:
        out['mucape'] = np.where(
            np.maximum(0.0, mucape_i) >= MUCAPE_MIN,
            np.maximum(0.0, mucape_i), 0.0,
        ).astype(np.float32)
        log.info(f'[mucape] max={float(out["mucape"].max()):.0f} J/kg '
                 f'active={int((out["mucape"] > 0).sum())} cells')
    else:
        log.warning('blend: mucape skipped (RAP MUCAPE not available)')

    # --- LCL height: log-form approximation using RTMA T/Td ---------------
    # Computed here (before SRH) so z0 can use LCL as the mixed-layer
    # depth proxy, and so STP can reference lcl without recomputing.
    # The classic Espy 125×depression breaks down at small (<2K) or large
    # (>15K) dewpoint depressions. The denominator form is numerically better:
    #   LCL = (T_C − Td_C) / (0.0012 + 0.00012 × T_C)
    # Denominator zeroes at T_C = −10°C; clamp to 1e-4 to avoid division error.
    # Uses TPW-corrected td2m if tpw_data was provided above.
    _t2m_c  = t2m  - 273.15
    _td2m_c = td2m - 273.15
    _lcl_denom = np.maximum(0.0012 + 0.00012 * _t2m_c, 1e-4)
    lcl = (_t2m_c - _td2m_c) / _lcl_denom   # meters AGL
    out['lcl'] = np.clip(lcl, 0.0, 4000.0).astype(np.float32)

    # --- 0-3km lapse rate: (T_sfc − T_700) / ~3km --------------------------
    # 700mb ≈ 3000m MSL; using T_sfc (RTMA K) − T_700 (RAP K) as a proxy for
    # the 0-3km environmental lapse rate. Strong lapse rates (≥7 K/km) combined
    # with significant SRH indicate a classic tornado setup even when CAPE is
    # modest. No RTMA thermal correction on T_700 — the sfc T correction
    # already propagates into sbcape/STP; correcting both would double-count.
    if t700_i is not None:
        lr_km = (t2m - t700_i) / 3.0          # K/km, positive = unstable
        out['lapse3km'] = np.clip(lr_km, 0.0, 12.0).astype(np.float32)
        log.info(f'[lapse3km] max={float(out["lapse3km"].max()):.1f} K/km '
                 f'active={int((out["lapse3km"] >= 7.0).sum())} cells ≥7 K/km')
    else:
        log.warning('blend: lapse3km skipped (t700 missing)')

    # --- 0-1km SRH: exponential-decay RTMA wind correction ---------------
    # Method:
    #   1. Surface anomaly: RTMA 10m − RAP 10m
    #   2. Exponential decay through BL: V_corr(z) = V_RAP(z) + Δ·exp(−z/z0)
    #      z0 = clip(LCL, 400, 1200m) — deeper mixed layers allow the surface
    #      anomaly to propagate higher; weights vary spatially with LCL.
    #   3. 4-layer hodograph integral: surface(RTMA) → 950mb → 925mb → 850mb
    #      Falls back to 2-layer (sfc→925→850) if 950mb unavailable.
    #   4. RAP native USTM/VSTM for storm motion (no manual Bunkers)
    #   5. 50/50 blend with RAP native HLCY as backstop

    if (u850_i is not None and v850_i is not None and
            ustm_i is not None and vstm_i is not None):

        # Dynamic BL scale height: use LCL as proxy for mixed layer depth.
        # Deep mixed layers (High Plains summer) → larger z0 → anomaly
        # propagates higher. Shallow marine layers → smaller z0 → anomaly
        # stays near surface. Clamp to physically realistic range [400, 1200m].
        z0 = np.clip(lcl, 400.0, 1200.0)

        du_anom = u10 - u10_rap if u10_rap is not None else np.zeros_like(u10)
        dv_anom = v10 - v10_rap if v10_rap is not None else np.zeros_like(v10)

        u850_corr = u850_i + du_anom * np.exp(-1500.0 / z0)   # w ≈ 0.05
        v850_corr = v850_i + dv_anom * np.exp(-1500.0 / z0)

        if (u950_i is not None and v950_i is not None and
                u925_i is not None and v925_i is not None):
            # 4-layer: surface(10m) → 950mb(~500m) → 925mb(~750m) → 850mb(~1500m)
            u950_corr = u950_i + du_anom * np.exp(-500.0  / z0)   # w ≈ 0.37
            v950_corr = v950_i + dv_anom * np.exp(-500.0  / z0)
            u925_corr = u925_i + du_anom * np.exp(-750.0  / z0)   # w ≈ 0.22
            v925_corr = v925_i + dv_anom * np.exp(-750.0  / z0)

            # Layer 1: surface(10m) → 950mb (~500m AGL)
            srh_l1 = ((u10       - ustm_i) * (v950_corr - v10)       -
                      (v10       - vstm_i) * (u950_corr - u10))
            # Layer 2: 950mb → 925mb (~500m → 750m AGL)
            srh_l2 = ((u950_corr - ustm_i) * (v925_corr - v950_corr) -
                      (v950_corr - vstm_i) * (u925_corr - u950_corr))
            # Layer 3: 925mb → 850mb (~750m → 1500m AGL)
            srh_l3 = ((u925_corr - ustm_i) * (v850_corr - v925_corr) -
                      (v925_corr - vstm_i) * (u850_corr - u925_corr))

            srh_rtma = np.clip(np.abs(srh_l1 + srh_l2 + srh_l3), 0.0, 1200.0).astype(np.float32)
            log.info(f'[srh1] 4-layer: sfc→950→925→850mb, raw_max={float(srh_rtma.max()):.0f}')

        elif u925_i is not None and v925_i is not None:
            # 2-layer fallback: surface(10m) → 925mb → 850mb
            u925_corr = u925_i + du_anom * np.exp(-750.0  / z0)
            v925_corr = v925_i + dv_anom * np.exp(-750.0  / z0)

            srh_l1 = ((u10       - ustm_i) * (v925_corr - v10)       -
                      (v10       - vstm_i) * (u925_corr - u10))
            srh_l2 = ((u925_corr - ustm_i) * (v850_corr - v925_corr) -
                      (v925_corr - vstm_i) * (u850_corr - u925_corr))

            srh_rtma = np.clip(np.abs(srh_l1 + srh_l2), 0.0, 1200.0).astype(np.float32)
            log.info(f'[srh1] 2-layer fallback (950mb unavail): sfc→925→850mb, '
                     f'raw_max={float(srh_rtma.max()):.0f}')

        else:
            # Minimal 1-layer fallback: surface → 850mb only
            srh_l1 = ((u10    - ustm_i) * (v850_corr - v10) -
                      (v10    - vstm_i) * (u850_corr - u10))
            srh_rtma = np.clip(np.abs(srh_l1), 0.0, 1200.0).astype(np.float32)
            log.info(f'[srh1] 1-layer fallback (925/950mb unavail): sfc→850mb, '
                     f'raw_max={float(srh_rtma.max()):.0f}')

        if hrrr_srh_i is not None and srh1_i is not None:
            # 3-way blend: RTMA-corrected 40%, HRRR 3km native 40%, RAP 13km 20%.
            # HRRR captures mesoscale SRH gradients near boundaries better than
            # RAP (3km vs 13km); the RTMA component adds the surface wind
            # correction that neither model has below 3km resolution.
            out['srh1'] = (0.4 * srh_rtma +
                           0.4 * hrrr_srh_i +
                           0.2 * srh1_i).astype(np.float32)
            log.info(f'[srh1] 3-way blend (RTMA 40% + HRRR 40% + RAP 20%): '
                     f'max={float(out["srh1"].max()):.0f} '
                     f'rtma_max={float(srh_rtma.max()):.0f} '
                     f'hrrr_max={float(hrrr_srh_i.max()):.0f} '
                     f'rap_max={float(srh1_i.max()):.0f}')

        elif hrrr_srh_i is not None:
            # HRRR available but no RAP HLCY: 50/50 RTMA + HRRR
            out['srh1'] = (0.5 * srh_rtma + 0.5 * hrrr_srh_i).astype(np.float32)
            log.info(f'[srh1] RTMA+HRRR (no RAP HLCY): '
                     f'max={float(out["srh1"].max()):.0f} '
                     f'rtma_max={float(srh_rtma.max()):.0f} '
                     f'hrrr_max={float(hrrr_srh_i.max()):.0f}')

        elif srh1_i is not None:
            # No HRRR: original 50/50 RTMA + RAP
            out['srh1'] = (0.5 * srh_rtma + 0.5 * srh1_i).astype(np.float32)
            log.info(f'[srh1] 50/50 blend RTMA+RAP (HRRR unavailable): '
                     f'max={float(out["srh1"].max()):.0f} '
                     f'rtma_max={float(srh_rtma.max()):.0f} '
                     f'rap_max={float(srh1_i.max()):.0f}')

        else:
            # Fallback: RTMA-only
            out['srh1'] = srh_rtma
            log.info(f'[srh1] RTMA-only (all backstops missing): '
                     f'max={float(srh_rtma.max()):.0f}')

    elif hrrr_srh_i is not None and srh1_i is not None:
        out['srh1'] = (0.5 * hrrr_srh_i + 0.5 * srh1_i).astype(np.float32)
        log.warning('[srh1] HRRR+RAP blend (no RTMA correction — ustm/vstm missing)')
    elif hrrr_srh_i is not None:
        out['srh1'] = hrrr_srh_i
        log.warning('[srh1] HRRR-only (no RTMA correction, no RAP HLCY)')
    elif srh1_i is not None:
        out['srh1'] = srh1_i
        log.warning('[srh1] RAP-only (ustm/vstm missing, HRRR unavailable)')
    else:
        log.warning('blend: srh1 skipped (all SRH sources missing)')

    # --- 0-6km BWD: pure bulk vector difference (V500mb − V10m_RTMA) ---------
    # Breaking into sub-layers and summing magnitudes computes total hodograph
    # length (path), not bulk shear. A curved hodograph inflates sub-layer sums
    # far above the true bulk vector, driving the STP shear term to its 1.5 cap
    # everywhere and making it useless as a discriminator.
    # Fix: single pure vector difference per Thompson et al. (2003) definition.
    BWD6_MIN = 15.0   # m/s — display gate
    if u500_i is not None and v500_i is not None:
        raw = np.sqrt((u500_i - u10)**2 + (v500_i - v10)**2).astype(np.float32)
        out['bwd6'] = np.where(raw >= BWD6_MIN, raw, 0.0).astype(np.float32)
        log.info(f'bwd6: pure vector |V500−V10_RTMA|, '
                 f'gate={BWD6_MIN}m/s, active={int((raw >= BWD6_MIN).sum())} cells')
    else:
        log.warning('blend: bwd6 skipped (u500/v500 missing)')

    # --- CIN-gated STP (Thompson et al. 2003 + 2012 CIN gate) -----------
    # CIN gate removed temporarily for diagnostic purposes.
    if 'sbcape' in out and 'srh1' in out and 'bwd6' in out:
        cape_term  = out['sbcape'] / 1500.0
        lcl_term   = np.clip((2000.0 - lcl) / 1000.0, 0.0, 1.0).astype(np.float32)
        srh_term   = out['srh1'] / 150.0
        shear_term = np.minimum(1.5, out['bwd6'] / 20.0)

        raw_stp = cape_term * lcl_term * srh_term * shear_term
        out['stp'] = np.nan_to_num(
            np.maximum(0, raw_stp), nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)
        log.info(f'[stp] max={float(out["stp"].max()):.2f} (no CIN gate)')
    else:
        log.warning('blend: stp skipped (sbcape, srh1, or bwd6 missing)')

    # --- Supercell Composite Parameter (SCP) ----------------------------
    # SCP = (SBCAPE/1000) * (SRH1/50) * (BWD6/20)
    # Gate: BWD6 < 10 m/s → SCP = 0 (no kinematic organization)
    # Gate: SBCAPE < 100 J/kg → SCP = 0 (no thermodynamic support)
    # Normalizations per SPC operational SCP documentation.
    # SCP highlights where rotating storms are favored BEFORE tornado
    # potential — complements STP which focuses on tornado environments.
    if 'sbcape' in out and 'srh1' in out and 'bwd6' in out:
        scp_raw = (out['sbcape'] / 1000.0) * (out['srh1'] / 50.0) * (out['bwd6'] / 20.0)
        scp_raw = np.where(out['bwd6']   <  10.0, 0.0, scp_raw)
        scp_raw = np.where(out['sbcape'] < 100.0, 0.0, scp_raw)
        out['scp'] = np.nan_to_num(
            np.maximum(0, scp_raw), nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)
        log.info(f'[scp] max={float(out["scp"].max()):.2f}')
    else:
        log.warning('blend: scp skipped (sbcape, srh1, or bwd6 missing)')

    # --- EHI: Energy-Helicity Index -------------------------------------
    # EHI = (SBCAPE × 0-1km SRH) / 160000  (Thompson et al. 1998).
    # EHI > 1: rotating thunderstorm favored.
    # EHI > 2.5: significant tornado environment.
    # Gate: SBCAPE < 100 J/kg or SRH1 < 25 m²/s² → EHI = 0.
    if 'sbcape' in out and 'srh1' in out:
        ehi_raw = (out['sbcape'] * out['srh1']) / 160000.0
        ehi_raw = np.maximum(0.0, ehi_raw)
        ehi_raw = np.where(out['sbcape'] < 100.0, 0.0, ehi_raw)
        ehi_raw = np.where(out['srh1']   <  25.0, 0.0, ehi_raw)
        out['ehi'] = np.nan_to_num(
            ehi_raw, nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)
        log.info(f'[ehi] max={float(out["ehi"].max()):.2f} '
                 f'active={int((out["ehi"] > 0).sum())} cells')
    else:
        log.warning('blend: ehi skipped (sbcape or srh1 missing)')

    # --- Surface relative vorticity and convergence ----------------------
    # Smooth RTMA winds before differencing to suppress sub-mesoscale
    # observation noise from individual stations (buildings, trees, gusts).
    # sigma=1.5 at 2.5km spacing → ~4km effective smoothing — preserves
    # mesoscale features (fronts, outflows) while removing station noise.
    from scipy.ndimage import gaussian_filter
    u10_smooth = gaussian_filter(u10.astype(np.float64), sigma=1.5).astype(np.float32)
    v10_smooth = gaussian_filter(v10.astype(np.float64), sigma=1.5).astype(np.float32)

    # numpy.gradient on 2D array returns [d/dy, d/dx] for axis 0 and 1.
    du_dy = np.gradient(u10_smooth, RTMA_DY, axis=0)
    du_dx = np.gradient(u10_smooth, RTMA_DX, axis=1)
    dv_dy = np.gradient(v10_smooth, RTMA_DY, axis=0)
    dv_dx = np.gradient(v10_smooth, RTMA_DX, axis=1)

    # Scale to 10^-5 s^-1 (standard operational vorticity units).
    # Raw values at 2.5km grid are ~1e-4 s^-1 for strong fronts;
    # multiplying by 1e5 brings them to ~10 in display units.
    VORT_SCALE = 1e5

    # Cyclonic vorticity (positive = counterclockwise in N. hemisphere)
    vort_raw = (dv_dx - du_dy) * VORT_SCALE
    out['vort'] = np.where(vort_raw >= 2.0, vort_raw, 0.0).astype(np.float32)

    # Convergence: positive = inflow
    conv_raw = (-(du_dx + dv_dy)) * VORT_SCALE
    out['conv'] = np.where(conv_raw >= 2.0, conv_raw, 0.0).astype(np.float32)

    log.info(f'vort: max={float(out["vort"].max()):.1f}×10⁻⁵ s⁻¹ '
             f'active={int((out["vort"] > 0).sum())} cells')
    log.info(f'conv: max={float(out["conv"].max()):.1f}×10⁻⁵ s⁻¹ '
             f'active={int((out["conv"] > 0).sum())} cells')

    # --- SRV: Streamwise horizontal vorticity from 0-1km wind shear ------
    # Horizontal vorticity vector (from vertical wind shear, 10m → 850mb):
    #   ωx = −∂v/∂z ≈ −(v850 − v10) / 1500m
    #   ωy =  ∂u/∂z ≈  (u850 − u10) / 1500m
    # Streamwise component = projection onto storm-relative wind direction:
    #   SRV = (ωx·u_sr + ωy·v_sr) / |SR_wind|
    # Physical interpretation: positive SRV → horizontal vorticity tilts
    # into vertical rotation by the updraft (Rasmussen & Davies-Jones 1982).
    # Scale: output in 10^-3 s^-1; strong environments yield 5–20.
    # Uses u10_smooth/v10_smooth (already computed above for vort/conv).
    if (u850_i is not None and v850_i is not None and
            ustm_i is not None and vstm_i is not None):
        # Horizontal vorticity from 10m → 850mb layer (Δz ≈ 1500m)
        omega_x = -(v850_i - v10_smooth) / 1500.0   # s⁻¹
        omega_y =  (u850_i - u10_smooth) / 1500.0   # s⁻¹

        # Storm-relative wind at surface (unit vector)
        u_sr   = u10_smooth - ustm_i
        v_sr   = v10_smooth - vstm_i
        sr_spd = np.maximum(np.sqrt(u_sr**2 + v_sr**2), 1e-3)

        # Streamwise projection
        srv_raw = (omega_x * u_sr + omega_y * v_sr) / sr_spd   # s⁻¹

        # Keep only positive (cyclonically favoured) values.
        # SRV from vertical wind shear (10m→850mb, Δz≈1500m) is inherently
        # ~100× larger than surface relative vorticity from horizontal gradients.
        # Scale by 1e3 → display values in ×10⁻³ s⁻¹; strong environments
        # yield 5–20, fitting the 0–30 iOS range. The legend label reflects this.
        SRV_SCALE = 1e3
        srv_scaled = np.maximum(0.0, srv_raw) * SRV_SCALE
        out['srv'] = np.where(srv_scaled >= 2.0, srv_scaled, 0.0).astype(np.float32)
        log.info(f'[srv] max={float(out["srv"].max()):.1f}×10⁻³ s⁻¹ '
                 f'active={int((out["srv"] > 0).sum())} cells')
    else:
        log.warning('blend: srv skipped (u850/v850 or ustm/vstm missing)')

    # --- Td depression: T - Td (K) ---------------------------------------
    # Lower values = more moist; useful as a dryline proxy.
    # Uses TPW-corrected td2m if tpw_data was provided above.
    out['td_dep'] = (t2m - td2m).astype(np.float32)

    # --- BACI: Bay Area Convection Index --------------------------------
    # Regional parameter — non-zero only within lat 36.5–38.5, lon -123 to -120.
    out['baci'] = compute_baci(
        rtma=rtma, rap=rap,
        rap_lats=rap_lats, rap_lons=rap_lons,
        rtma_lats=rtma_lats, rtma_lons=rtma_lons,
    )

    # Pass grid coordinates through for writer.py bbox calculation
    out['lats'] = rtma_lats
    out['lons'] = rtma_lons

    ready = [k for k in out if k not in ('lats', 'lons')]
    log.info(f'Blend complete: {ready}')
    return out
