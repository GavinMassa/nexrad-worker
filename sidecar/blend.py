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


def _rtma_scale_factor(lats_deg: np.ndarray) -> np.ndarray:
    """
    Lambert Conformal Conic map scale factor for the RTMA 2.5km grid.
    RTMA LCC parameters: lon0=-95°, lat0=25°, lat1=25°, lat2=25° (single SP).
    m(lat) = cos(lat1) / cos(lat) * (tan(π/4 + lat/2) / tan(π/4 + lat1/2))^n
    where n = sin(lat1) for a tangent cone (lat1=lat2).
    Returns array same shape as lats_deg. Values ~0.86 at 50°N, ~1.0 at 25°N.
    """
    lat1_rad = np.radians(25.0)   # RTMA standard parallel
    n = np.sin(lat1_rad)           # cone constant for tangent LCC
    lat_rad = np.radians(lats_deg)
    # Ratio of map distances at lat vs. standard parallel
    scale = (np.cos(lat1_rad) / np.cos(lat_rad)) * \
            (np.tan(np.pi / 4 + lat_rad / 2) /
             np.tan(np.pi / 4 + lat1_rad / 2)) ** n
    return scale.astype(np.float32)


def compute_cape_cin_lifted(
    t2m:    np.ndarray,   # RTMA 2m temperature (K), shape (ny, nx)
    td2m:   np.ndarray,   # RTMA 2m dewpoint (K), TPW-corrected, shape (ny, nx)
    levels: list,         # list of (p_mb: float, T: np.ndarray|None, Td: np.ndarray|None)
                          #   sorted descending by pressure (lowest altitude last)
                          #   e.g. [(950, t950_i, td950_i), (925, ...), ...]
    ml_depth_mb: float = 100.0,   # mixed-layer averaging depth (mb) for MLCAPE
) -> tuple:
    """
    Compute SBCAPE, SBCIN, and MLCAPE via pseudoadiabatic parcel lift.

    Surface parcel: RTMA t2m / td2m (obs-anchored).
    Upper profile:  RRFS pressure-level T and Td fields (already interpolated
                    to the RTMA 2.5km grid).
    Integration:    trapezoidal rule on virtual-temperature buoyancy.

    Returns (sbcape, sbcin, mlcape) — each shape (ny, nx), dtype float32,
    non-negative for CAPE, non-positive for CIN (clipped to -300 J/kg).
    """
    from scipy.ndimage import gaussian_filter  # noqa: F401 (imported for consistency)

    # ── Physical constants ────────────────────────────────────────────────────
    Rd   = 287.04    # J/(kg·K)  dry air gas constant
    Rv   = 461.5     # J/(kg·K)  water vapour gas constant
    g    = 9.80665   # m/s²
    Lv   = 2.501e6   # J/kg      latent heat of vaporisation (0°C)
    cp   = 1005.7    # J/(kg·K)  specific heat of dry air at const pressure
    eps  = Rd / Rv   # ≈ 0.622

    # ── Scalar helpers (operate element-wise on arrays) ───────────────────────
    def _es(T_K: np.ndarray) -> np.ndarray:
        """Saturation vapour pressure (hPa) via Bolton (1980)."""
        tc = np.clip(T_K - 273.15, -80.0, 60.0)   # clamp: -80°C avoids exp overflow, 60°C physical max
        return 6.112 * np.exp(17.67 * tc / (tc + 243.5))

    def _mixr(Td_K: np.ndarray, p_mb: float) -> np.ndarray:
        """Mixing ratio (kg/kg) from dewpoint (K) and pressure (mb/hPa)."""
        e = _es(Td_K)          # hPa
        return eps * e / np.maximum(p_mb - e, 1.0)

    def _Tv(T_K: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Virtual temperature (K)."""
        return T_K * (1.0 + w / eps) / (1.0 + w)

    def _wobus(T_K: np.ndarray) -> np.ndarray:
        """
        Wobus polynomial: moist-adiabatic lapse rate correction (K).
        Based on Wobus (1966) as implemented in NSHARP/SHARPpy.
        Input: temperature (K).
        """
        tc = T_K - 273.15
        x  = tc - 20.0
        # Two polynomial segments joined at x = 0 (tc = 20 °C)
        pol1 = (((((((( 2.0103e-9  * x) +
                        (-1.6665e-7) ) * x +
                        5.8464e-6  ) * x +
                       -7.3765e-5  ) * x +
                       4.8534e-4   ) * x +
                      -2.9407e-3   ) * x +
                       1.8060e-2   ) * x +
                      -7.6617e-2   ) * x + \
                      -7.8581e-3
        pol2 = ((((((((-1.5966e-9  * x) +
                        1.5422e-7   ) * x +
                       -2.2530e-6  ) * x +
                       -1.2978e-5  ) * x +
                       1.5542e-4   ) * x +
                      -6.3523e-4   ) * x +
                       1.8927e-2   ) * x +
                       1.5705e-1   ) * x + \
                      -7.8581e-3
        return np.where(x >= 0, pol1, pol2)

    def _moist_adiabat_dT(T_K: np.ndarray, p_mb: float) -> np.ndarray:
        """
        dT/dp for a saturated (moist) adiabat (K/mb).
        Positive => temperature increases with pressure (downward).
        """
        es_val = _es(T_K)                                           # hPa
        w_s    = eps * es_val / np.maximum(p_mb - es_val, 1.0)    # p in hPa, matches es
        numer  = (Rd * T_K / (p_mb * 100.0)) * (1.0 + Lv * w_s / (Rd * T_K))  # p in Pa
        denom  = cp + Lv**2 * w_s / (Rv * T_K**2)
        return numer / denom * 100.0   # K/Pa → K/mb

    def _lift_parcel_to_level(
        T_parcel:    np.ndarray,
        w_parcel:    np.ndarray,   # mixing ratio at LCL (kg/kg), constant above
        p_start:     float,
        p_end:       float,
        is_below_lcl: np.ndarray,  # bool mask: still dry adiabatic lift?
        lcl_p:       np.ndarray,   # LCL pressure (mb), per gridpoint
    ) -> tuple:
        """
        Lift a parcel from p_start to p_end using small finite steps.
        Below the LCL: dry adiabatic (Γd = g/cp = 9.77 K/km).
        Above the LCL: moist pseudoadiabatic using Wobus correction.
        Returns (T_parcel_at_p_end, still_below_lcl_mask).
        """
        _p_start_s = float(np.mean(p_start)) if isinstance(p_start, np.ndarray) else float(p_start)
        n_steps = max(1, int(abs(p_end - _p_start_s) / 5.0))  # 5 mb sub-steps
        dp = (p_end - p_start) / n_steps   # may be ndarray if p_start is 2D
        T_p   = T_parcel.copy()
        below = is_below_lcl.copy()
        p_cur = _p_start_s
        for _ in range(n_steps):
            p_mid = p_cur + dp / 2.0
            # Dry-adiabatic dT/dp = Rd*T/(cp*p)  [K/mb, signed by direction]
            dT_dry  = (Rd / cp) * T_p / p_mid * dp
            # Moist-adiabatic step
            dT_moist = _moist_adiabat_dT(T_p, p_mid) * dp
            T_p   = np.where(below, T_p + dT_dry, T_p + dT_moist)
            p_cur = p_cur + dp
            below = below & (p_cur > lcl_p)  # cross LCL during this step?
        return T_p, below

    # ── Surface parcel properties ─────────────────────────────────────────────
    # Surface pressure: use 1000mb as CONUS mean sea-level equivalent.
    # 975mb was too low — it placed the parcel origin 25mb above the actual
    # surface on the central plains (~300m MSL), causing the dry-adiabatic
    # cooling to overshoot the cap and never find the LFC.
    P_SFC    = 1000.0
    t2m  = np.clip(t2m,  200.0, 330.0)
    td2m = np.clip(td2m, 180.0, 320.0)
    w_sfc    = _mixr(td2m, P_SFC)
    Tv_sfc   = _Tv(t2m, w_sfc)

    # Bolton (1980) LCL temperature and pressure
    # T_lcl = 1 / (1/(Td-56) + ln(T/Td)/800) + 56   (all in K)
    lcl_T = 1.0 / (1.0 / np.maximum(td2m - 56.0, 0.5) +
                   np.log(np.maximum(t2m / np.maximum(td2m, 200.0), 1.0)) / 800.0
                   ) + 56.0
    lcl_p = P_SFC * (lcl_T / t2m) ** (cp / Rd)  # Poisson: p_lcl / p_sfc

    # ── Build combined level arrays (surface + upper-air) ────────────────────
    # levels is sorted descending in pressure (highest p first),
    # e.g. [(950, t950_i, td950_i), (925,...), (850,...), (700,...), (500,...)]
    # Prepend surface so integration starts from there.
    all_p  = [P_SFC] + [lv[0] for lv in levels]
    all_T  = [t2m]   + [lv[1] for lv in levels]
    all_Td = [td2m]  + [lv[2] for lv in levels]

    # ── Inner integration function ────────────────────────────────────────────
    def _integrate_cape_cin(
        parcel_T_sfc:  np.ndarray,   # parcel T at surface (K)
        parcel_Td_sfc: np.ndarray,   # parcel Td at surface (K)
        p_sfc:         float,
    ) -> tuple:
        """
        Lift the parcel defined by (parcel_T_sfc, parcel_Td_sfc) from p_sfc
        through all_p/all_T/all_Td and integrate CAPE / CIN.
        Returns (cape_arr, cin_arr) — shape (ny, nx), float32.
        """
        w_p    = _mixr(parcel_Td_sfc, p_sfc)
        lcl_p_ = p_sfc * (_lift_parcel_to_level.__defaults__[0] if False
                           else (1.0 / np.maximum(parcel_Td_sfc - 56.0, 0.5) +
                                 np.log(np.maximum(parcel_T_sfc /
                                                   np.maximum(parcel_Td_sfc, 200.0), 1.0)
                                        ) / 800.0) ** -1 + 56.0)
        # Cleaner: just recompute LCL pressure for this parcel
        lcl_T_ = 1.0 / (1.0 / np.maximum(parcel_Td_sfc - 56.0, 0.5) +
                         np.log(np.maximum(parcel_T_sfc /
                                           np.maximum(parcel_Td_sfc, 200.0), 1.0)
                                ) / 800.0) + 56.0
        lcl_p_ = p_sfc * (lcl_T_ / parcel_T_sfc) ** (cp / Rd)

        T_p   = parcel_T_sfc.copy()
        below = np.ones(T_p.shape, dtype=bool)      # all below LCL initially
        cape  = np.zeros(T_p.shape, dtype=np.float64)
        cin   = np.zeros(T_p.shape, dtype=np.float64)

        # Environment Tv at surface: use RTMA td2m (from outer scope) for env moisture
        w_e_sfc   = _mixr(td2m, p_sfc if np.isscalar(p_sfc) else float(np.mean(p_sfc)))
        Tv_e_prev = _Tv(t2m, w_e_sfc)
        Tv_p_prev = _Tv(T_p, w_p)

        for i in range(1, len(all_p)):
            p_prev = all_p[i - 1]
            p_cur  = float(all_p[i]) if not isinstance(all_p[i], np.ndarray) else all_p[i]
            # p_prev may be ndarray (surface) or float (upper levels); _lift handles both
            T_e_cur  = all_T[i]
            Td_e_cur = all_Td[i]
            if T_e_cur is None or Td_e_cur is None:
                continue                              # skip missing levels
            # Lift parcel from previous level to current
            T_p, below = _lift_parcel_to_level(T_p, w_p, p_prev, p_cur,
                                                below, lcl_p_)
            w_e_cur  = _mixr(Td_e_cur, p_cur)
            Tv_e_cur = _Tv(T_e_cur, w_e_cur)
            Tv_p_cur = _Tv(T_p, w_p)

            # Trapezoidal buoyancy integral via hypsometric equation:
            #   CAPE/CIN = ∫ g·(Tv_p - Tv_e)/Tv_e · dz
            #   dz = -(Rd/g)·Tv_e · d(ln p)              [hypsometric]
            # Substituting:
            #   CAPE/CIN = ∫ g·(Tv_p - Tv_e)/Tv_e · (-(Rd/g)·Tv_e) · d(ln p)
            #            = ∫ -Rd·(Tv_p - Tv_e) · d(ln p)
            #            = Rd·(Tv_p - Tv_e) · ln(p_prev/p_cur)   [trapezoidal, ascending]
            # The Tv_e denominators cancel — use raw ΔTv (K), not the dimensionless ratio.
            # Using the ratio (Tv_p-Tv_e)/Tv_e would produce values ~270× too small.
            dT_l    = Tv_p_prev - Tv_e_prev   # virtual-temperature difference (K)
            dT_r    = Tv_p_cur  - Tv_e_cur
            p_prev_safe = np.maximum(p_prev, 1.0) if isinstance(p_prev, np.ndarray) else max(p_prev, 1.0)
            p_cur_safe  = float(p_cur) if not isinstance(p_cur, np.ndarray) else np.maximum(p_cur, 1.0)
            dA      = Rd * (dT_l + dT_r) / 2.0 * np.log(p_prev_safe / p_cur_safe)
            # ln(p_prev/p_cur) > 0 (ascending); dA > 0 when parcel warmer than env.
            lfc_found = (cape > 0)          # any prior positive buoyancy = above LFC
            cin  = cin  + np.where(~lfc_found & (dA < 0), dA, 0.0)
            cape = cape + np.where(dA > 0, dA, 0.0)

            Tv_e_prev = Tv_e_cur
            Tv_p_prev = Tv_p_cur

        return cape.astype(np.float32), np.maximum(cin, -300.0).astype(np.float32)

    # ── SBCAPE / SBCIN — surface parcel ──────────────────────────────────────
    sbcape, sbcin = _integrate_cape_cin(t2m, td2m, P_SFC)

    # ── MLCAPE — mean-layer parcel (lowest ml_depth_mb of the atmosphere) ────
    ml_p_top = float(np.mean(P_SFC)) - ml_depth_mb    # scalar — P_SFC is now 2D; mean keeps loop comparison unambiguous
    ml_T_sum  = t2m.copy().astype(np.float64)
    ml_Td_sum = td2m.copy().astype(np.float64)
    ml_n      = np.ones(t2m.shape, dtype=np.float64)   # surface already counted

    for p_mb, T_lev, Td_lev in levels:
        if T_lev is not None and Td_lev is not None and p_mb >= ml_p_top:
            ml_T_sum  = ml_T_sum  + T_lev.astype(np.float64)
            ml_Td_sum = ml_Td_sum + Td_lev.astype(np.float64)
            ml_n      = ml_n + 1.0

    ml_T_mean  = (ml_T_sum  / ml_n).astype(np.float32)
    ml_Td_mean = (ml_Td_sum / ml_n).astype(np.float32)
    mlcape, _  = _integrate_cape_cin(ml_T_mean, ml_Td_mean, P_SFC)

    return sbcape, sbcin, mlcape


def blend(rtma: dict, upper_air: dict, tpw_data: dict | None = None,
          hrrr_hlcy: dict | None = None) -> dict:
    """
    Blend RTMA 2.5km surface fields with upper-air model fields.

    rtma:       output of fetch_rtma    — t2m, td2m, u10, v10, lats, lons
    upper_air:  output of fetch_rrfs (or fetch_rap) — cape, cin, srh1,
                u500, v500, u850, v850, u925, v925, u950, v950,
                ustm, vstm, mucape, cape3k, t700, t925, pwat, lats_rap, lons_rap
    tpw_data:   output of fetch_tpw    — tpw, lats, lons (float32 2D arrays); or None
    hrrr_hlcy:  output of fetch_hrrr_hlcy — hlcy, lats, lons (float32 2D arrays); or None
                HRRR 0-1km SRH at native 3km resolution; used as second backstop
                in the SRH blend. Failure to provide this does not abort the blend.

    Returns dict of float32 grids on the RTMA 2.5km grid, plus lats/lons.
    Missing inputs produce a warning and the dependent params are omitted
    rather than crashing.
    """
    rtma_lats = rtma['lats']    # (1597, 2345)
    rtma_lons = rtma['lons']    # (1597, 2345)
    rap_lats  = upper_air['lats_rap']
    rap_lons  = upper_air['lons_rap']

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
    log.info('Interpolating upper-air fields to RTMA 2.5km grid...')
    cape_i   = interp(upper_air.get('cape'),     'cape')
    cin_i    = interp(upper_air.get('cin'),      'cin')
    srh1_i   = interp(upper_air.get('srh1'),     'srh1')
    u500_i   = interp(upper_air.get('u500'),     'u500')
    v500_i   = interp(upper_air.get('v500'),     'v500')
    u850_i   = interp(upper_air.get('u850'),     'u850')
    v850_i   = interp(upper_air.get('v850'),     'v850')
    u10_rap  = interp(upper_air.get('u10'),      'u10_rap')
    v10_rap  = interp(upper_air.get('v10'),      'v10_rap')
    u925_i   = interp(upper_air.get('u925'),     'u925')
    v925_i   = interp(upper_air.get('v925'),     'v925')
    u950_i   = interp(upper_air.get('u950'),     'u950')
    v950_i   = interp(upper_air.get('v950'),     'v950')
    ustm_i   = interp(upper_air.get('ustm'),     'ustm')
    vstm_i   = interp(upper_air.get('vstm'),     'vstm')
    t2m_rap  = interp(upper_air.get('t2m_rap'),  't2m_rap')
    td2m_rap = interp(upper_air.get('td2m_rap'), 'td2m_rap')
    cape3k_i = interp(upper_air.get('cape3k'),   'cape3k')
    mucape_i = interp(upper_air.get('mucape'),   'mucape')
    t700_i   = interp(upper_air.get('t700'),     't700')
    t850_i   = interp(upper_air.get('t850'),     't850')
    t500_i   = interp(upper_air.get('t500'),     't500')
    t950_i   = interp(upper_air.get('t950'),     't950')
    t925_i   = interp(upper_air.get('t925'),     't925')
    rh500_i  = interp(upper_air.get('rh500'),    'rh500')
    rh700_i  = interp(upper_air.get('rh700'),    'rh700')
    rh850_i  = interp(upper_air.get('rh850'),    'rh850')
    rh925_i  = interp(upper_air.get('rh925'),    'rh925')
    rh950_i  = interp(upper_air.get('rh950'),    'rh950')
    # Upper-troposphere levels — extend integration to the EL (~200 mb).
    t600_i   = interp(upper_air.get('t600'),     't600')
    rh600_i  = interp(upper_air.get('rh600'),    'rh600')
    t400_i   = interp(upper_air.get('t400'),     't400')
    rh400_i  = interp(upper_air.get('rh400'),    'rh400')
    t300_i   = interp(upper_air.get('t300'),     't300')
    rh300_i  = interp(upper_air.get('rh300'),    'rh300')
    t200_i   = interp(upper_air.get('t200'),     't200')
    rh200_i  = interp(upper_air.get('rh200'),    'rh200')

    def _rh_to_td(T_K: np.ndarray, rh: np.ndarray):
        """Convert RH (0–100) + T (K) to Td (K) via Bolton inverse."""
        if T_K is None or rh is None:
            return None
        rh_frac = np.clip(rh, 1.0, 100.0) / 100.0
        tc = np.clip(T_K - 273.15, -80.0, 60.0)
        ln_rh = np.log(rh_frac)
        ln_es  = 17.67 * tc / (tc + 243.5)
        td_c   = 243.5 * (ln_rh + ln_es) / (17.67 - ln_rh - ln_es)
        return (td_c + 273.15).astype(np.float32)

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

    # ── Obs-anchored SBCAPE / SBCIN / MLCAPE via lifted parcel ───────────────
    # Build pressure-level profile from RRFS fields (descending pressure order
    # so the shallowest levels are first). Derive Td from RH using Bolton inverse.
    td950_i = _rh_to_td(t950_i, rh950_i)
    td925_i = _rh_to_td(t925_i, rh925_i)
    td850_i = _rh_to_td(t850_i, rh850_i)
    td700_i = _rh_to_td(t700_i, rh700_i)
    td500_i = _rh_to_td(t500_i, rh500_i)
    td600_i = _rh_to_td(t600_i, rh600_i)
    td400_i = _rh_to_td(t400_i, rh400_i)
    td300_i = _rh_to_td(t300_i, rh300_i)
    td200_i = _rh_to_td(t200_i, rh200_i)

    # Levels in descending pressure order (surface → tropopause).
    # Upper levels (600→200 mb) are essential: the EL for strong convection
    # is at 200–250 mb, and the 500→200 mb layer contributes 1500–3000 J/kg.
    lift_levels = [
        (950.0, t950_i, td950_i),
        (925.0, t925_i, td925_i),
        (850.0, t850_i, td850_i),
        (700.0, t700_i, td700_i),
        (600.0, t600_i, td600_i),
        (500.0, t500_i, td500_i),
        (400.0, t400_i, td400_i),
        (300.0, t300_i, td300_i),
        (200.0, t200_i, td200_i),
    ]
    lift_levels = [(p, T, Td) for p, T, Td in lift_levels
                   if T is not None and Td is not None
                   and float(np.nanmax(T)) > 150.0]   # exclude zero-filled out-of-domain grids

    SBCAPE_MIN = 150.0   # J/kg — display gate

    if len(lift_levels) >= 2:
        sbcape_lifted, sbcin_lifted, mlcape_lifted = compute_cape_cin_lifted(
            t2m, td2m, lift_levels
        )
        raw_sbcape = np.maximum(0.0, sbcape_lifted)
        out['sbcape'] = np.where(
            raw_sbcape >= SBCAPE_MIN, raw_sbcape, 0.0
        ).astype(np.float32)
        out['sbcin'] = sbcin_lifted   # already clipped to [-300, 0]
        out['mlcape_lifted'] = np.where(
            mlcape_lifted >= SBCAPE_MIN, mlcape_lifted, 0.0
        ).astype(np.float32)
        log.info(f'[sbcape] lifted-parcel: raw_max={float(raw_sbcape.max()):.0f} '
                 f'gated_max={float(out["sbcape"].max()):.0f} '
                 f'active={int((out["sbcape"] > 0).sum())} cells '
                 f'(gate={SBCAPE_MIN:.0f} J/kg)')
        log.info(f'[sbcin] lifted-parcel: min={float(out["sbcin"].min()):.0f} J/kg')
        log.info(f'[mlcape] lifted-parcel: raw_max={float(np.maximum(0.0, mlcape_lifted).max()):.0f} '
                 f'gated_max={float(out["mlcape_lifted"].max()):.0f}')
        log.info(f'[cape_diag] t2m_sample={float(t2m.flat[t2m.size//2]):.1f}K '
                 f'td2m_sample={float(td2m.flat[td2m.size//2]):.1f}K '
                 f'mucape_ref={float(mucape_i.max()) if mucape_i is not None else "N/A":.0f}')
    else:
        # Fallback: fewer than 2 upper-air levels — use old linear correction
        log.warning('[sbcape] fewer than 2 RRFS pressure levels — falling back to '
                    'linear delta-T correction')
        if t2m_rap is not None:
            delta_t_clamped = np.clip(t2m - t2m_rap, -2.0, 2.0)
        else:
            delta_t_clamped = np.zeros_like(t2m)

        cape_i_raw = interp(upper_air.get('cape'), 'cape')
        cin_i_raw  = interp(upper_air.get('cin'),  'cin')

        if cape_i_raw is not None:
            sbcape_corrected = np.maximum(0, cape_i_raw + delta_t_clamped * 180.0)
            cape_mask        = cape_i_raw < 50.0
            raw_sbcape       = np.where(cape_mask, np.maximum(0, cape_i_raw),
                                        sbcape_corrected).astype(np.float32)
            out['sbcape']    = np.where(raw_sbcape >= SBCAPE_MIN,
                                        raw_sbcape, 0.0).astype(np.float32)
        if cin_i_raw is not None:
            cin_correction = delta_t_clamped * 40.0
            out['sbcin']   = np.clip(cin_i_raw + cin_correction,
                                     -300.0, 0.0).astype(np.float32)

    # --- 0-3km CAPE: low-level buoyancy from RRFS lev_0-3000_m layer --------
    # (unchanged — this is a model layer field, not surface-based)
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

    # --- MUCAPE: RRFS model most-unstable CAPE (180-0mb search layer) --------
    # Note: SBCAPE and MLCAPE now use the obs-anchored lifted parcel above.
    # MUCAPE retains the model value as it requires an elevated parcel search
    # that would need many more pressure levels to implement from scratch.
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
            # 3-way blend: RTMA-corrected 30%, HRRR 3km native 40%, RRFS 13km 30%.
            # HRRR captures mesoscale SRH gradients near boundaries best (3km);
            # RRFS improves on RAP in the mesoscale with hourly DA cycling;
            # RTMA adds the surface wind correction below 3km model resolution.
            out['srh1'] = (0.3 * srh_rtma +
                           0.4 * hrrr_srh_i +
                           0.3 * srh1_i).astype(np.float32)
            log.info(f'[srh1] 3-way blend (RTMA 30% + HRRR 40% + RRFS 30%): '
                     f'max={float(out["srh1"].max()):.0f} '
                     f'rtma_max={float(srh_rtma.max()):.0f} '
                     f'hrrr_max={float(hrrr_srh_i.max()):.0f} '
                     f'rrfs_max={float(srh1_i.max()):.0f}')

        elif hrrr_srh_i is not None:
            # HRRR available but no RRFS HLCY: 50/50 RTMA + HRRR
            out['srh1'] = (0.5 * srh_rtma + 0.5 * hrrr_srh_i).astype(np.float32)
            log.info(f'[srh1] RTMA+HRRR (no RRFS HLCY): '
                     f'max={float(out["srh1"].max()):.0f} '
                     f'rtma_max={float(srh_rtma.max()):.0f} '
                     f'hrrr_max={float(hrrr_srh_i.max()):.0f}')

        elif srh1_i is not None:
            # No HRRR: 50/50 RTMA + RRFS
            out['srh1'] = (0.5 * srh_rtma + 0.5 * srh1_i).astype(np.float32)
            log.info(f'[srh1] 50/50 blend RTMA+RRFS (HRRR unavailable): '
                     f'max={float(out["srh1"].max()):.0f} '
                     f'rtma_max={float(srh_rtma.max()):.0f} '
                     f'rrfs_max={float(srh1_i.max()):.0f}')

        else:
            # Fallback: RTMA-only
            out['srh1'] = srh_rtma
            log.info(f'[srh1] RTMA-only (all backstops missing): '
                     f'max={float(srh_rtma.max()):.0f}')

    elif hrrr_srh_i is not None and srh1_i is not None:
        out['srh1'] = (0.5 * hrrr_srh_i + 0.5 * srh1_i).astype(np.float32)
        log.warning('[srh1] HRRR+RRFS blend (no RTMA correction — ustm/vstm missing)')
    elif hrrr_srh_i is not None:
        out['srh1'] = hrrr_srh_i
        log.warning('[srh1] HRRR-only (no RTMA correction, no RRFS HLCY)')
    elif srh1_i is not None:
        out['srh1'] = srh1_i
        log.warning('[srh1] RRFS-only (ustm/vstm missing, HRRR unavailable)')
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
    # Compute once per cycle — shape matches RTMA grid (1597, 2345)
    m = _rtma_scale_factor(rtma_lats)

    # True physical spacing accounts for LCC map scale factor:
    # actual_dx(i,j) = RTMA_DX / m(i,j)
    # np.gradient with variable spacing requires 1D coordinate arrays,
    # so we apply the correction post-hoc by dividing by m.
    du_dy = np.gradient(u10_smooth, RTMA_DY, axis=0) * m
    du_dx = np.gradient(u10_smooth, RTMA_DX, axis=1) * m
    dv_dy = np.gradient(v10_smooth, RTMA_DY, axis=0) * m
    dv_dx = np.gradient(v10_smooth, RTMA_DX, axis=1) * m

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
    log.info(f'[vort/conv] LCC scale factor applied: '
             f'min={float(m.min()):.3f} max={float(m.max()):.3f}')

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
