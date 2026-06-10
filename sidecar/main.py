import asyncio, logging, json, gc, os
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from scipy.ndimage import zoom as ndimage_zoom
from fetch_rtma import fetch_rtma
from fetch_rap import fetch_rap, fetch_hrrr_hlcy
from fetch_rrfs import fetch_rrfs
from fetch_tpw import fetch_latest_tpw
from mesonet import fetch_mesonet_obs, compute_correction
from fetch_vad import fetch_vad_profiles, compute_site_layer_winds
from blend import blend as do_blend
from writer import write_output, OUT_DIR
from terrain_setup import build_terrain, terrain_already_exists
from satellite import satellite_worker

logging.basicConfig(level=logging.INFO, format='[sidecar] %(message)s')
log = logging.getLogger(__name__)

# Shared thread pool for CPU-bound work (IDW + blend derivation).
# max_workers=2 — both tasks are numpy-heavy; more workers contend on the
# same cores and RAM without benefit.
_thread_pool = ThreadPoolExecutor(max_workers=2)
_cycle_lock  = asyncio.Lock()

# Mesonet correction files written by mesonet_worker(), read by run_cycle().
# Stored in the same OUT_DIR (/app/sidecar-out) so they persist across restarts.
MESONET_DT_PATH    = OUT_DIR / 'mesonet_delta_t.bin'
MESONET_DTD_PATH   = OUT_DIR / 'mesonet_delta_td.bin'
MESONET_DPSFC_PATH = OUT_DIR / 'mesonet_delta_psfc.bin'
MESONET_META_PATH  = OUT_DIR / 'mesonet_meta.json'
MESONET_MAX_AGE_S  = 90 * 60   # discard corrections older than 90 minutes

# VAD observed-hodograph grids — written by vad_worker(), read by run_cycle().
# u/v interpolated to 500m and 1000m AGL + a coverage mask. blend.py combines
# these with the RTMA surface wind and model storm motion to form observed SRH.
VAD_U500_PATH  = OUT_DIR / 'vad_u500.bin'
VAD_V500_PATH  = OUT_DIR / 'vad_v500.bin'
VAD_U1000_PATH = OUT_DIR / 'vad_u1000.bin'
VAD_V1000_PATH = OUT_DIR / 'vad_v1000.bin'
VAD_COV_PATH   = OUT_DIR / 'vad_cov.bin'      # float32 (ny,nx): 1.0 where covered
VAD_META_PATH  = OUT_DIR / 'vad_meta.json'
VAD_MAX_AGE_S  = 45 * 60   # discard VAD grids older than 45 min


def log_memory(label: str = '') -> None:
    """Log current RSS so Railway logs show per-cycle memory trend."""
    mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    log.info(f'[mem] {label}: {mb:.0f} MB')


# ── Mesonet background worker ─────────────────────────────────────────────────

async def mesonet_worker():
    """
    Fetch IEM obs every 10 minutes, thin, run IDW on the 399×586 blend grid,
    write correction files atomically. Completely independent of run_cycle() —
    never blocks it and never crashes the process (all exceptions caught).

    Correction files are read by run_cycle() and applied in <1 s via a simple
    array read + upsample, with no IDW running inline with the blend pipeline.
    """
    FACTOR = 0.5

    while True:
        try:
            # Need meta.json to know the blend output grid dimensions and bbox.
            # Wait (non-blocking) until the first blend cycle has completed.
            meta_path = OUT_DIR / 'meta.json'
            if not meta_path.exists():
                log.info('[mesonet] waiting for first blend cycle to complete...')
                await asyncio.sleep(60)
                continue

            meta     = json.loads(meta_path.read_text())
            ny_small = meta['ny']
            nx_small = meta['nx']

            stations = await fetch_mesonet_obs()
            if not stations:
                log.warning('[mesonet] no obs — skipping correction update')
                await asyncio.sleep(600)
                continue

            # Restrict to the RTMA domain bbox from meta.json
            lat_min = meta['lat_min'];  lat_max = meta['lat_max']
            lon_min = meta['lon_min'];  lon_max = meta['lon_max']
            stations_domain = [
                s for s in stations
                if lat_min <= s['lat'] <= lat_max
                and lon_min <= s['lon'] <= lon_max
            ]
            log.info(f'[mesonet] {len(stations_domain)}/{len(stations)} in domain')

            # Spatially adaptive thinning:
            # East of -100° (dense ASOS/AWOS network): 0.75° spacing → ~2x station density
            # West of -100° (sparse network): 1.5° spacing → same as before
            # This sharpens moisture/temp gradients near boundaries in the Plains/East
            # without the memory cost of global densification.
            occupied = {}
            thinned = []
            for st in stations_domain:
                spacing = 0.75 if st['lon'] > -100.0 else 1.5
                key = (int(st['lat'] / spacing), int(st['lon'] / spacing), spacing)
                if key not in occupied:
                    occupied[key] = True
                    thinned.append(st)

            east_count = sum(1 for st in thinned if st['lon'] > -100.0)
            west_count = len(thinned) - east_count
            log.info(f'[mesonet] thinned {len(stations_domain)} → {len(thinned)} stations '
                     f'(east={east_count}@0.75°, west={west_count}@1.5°)')

            # Hard cap: IDW intermediate arrays scale as N_stations × N_gridpoints.
            # At 1350 stations on a 1597×2345 grid that's ~20GB — OOM on Railway Hobby.
            # Slice the front of the list; stations are already spatially distributed
            # by the thinning loop so the cap preserves geographic spread.
            MAX_STATIONS = 700
            if len(thinned) > MAX_STATIONS:
                thinned = thinned[:MAX_STATIONS]
                log.info(f'[mesonet] capped to {MAX_STATIONS} stations (memory limit)')

            if len(thinned) < 10:
                log.warning('[mesonet] too few stations after thinning — skipping')
                await asyncio.sleep(600)
                continue

            # Load real RTMA reference grids saved by run_cycle().
            # Using actual Lambert Conformal lats/lons (not linspace) ensures
            # KDTree nearest-gridpoint lookup finds the correct background value
            # at each station, preventing spurious QC rejections.
            lats_path = OUT_DIR / 'rtma_lats_small.bin'
            lons_path = OUT_DIR / 'rtma_lons_small.bin'
            t2m_path  = OUT_DIR / 'rtma_t2m_small.bin'
            td2m_path = OUT_DIR / 'rtma_td2m_small.bin'

            missing = [p for p in [lats_path, lons_path, t2m_path, td2m_path]
                       if not p.exists()]
            if missing:
                log.info(f'[mesonet] waiting for reference grids: '
                         f'{[p.name for p in missing]}')
                await asyncio.sleep(60)
                continue

            grid_lats = np.frombuffer(lats_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)
            grid_lons = np.frombuffer(lons_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)
            grid_t2m  = np.frombuffer(t2m_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)
            grid_td2m = np.frombuffer(td2m_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)

            # Barnes OA is CPU-bound but fast with the thinned network (~600 stations)
            loop = asyncio.get_running_loop()

            # Build background psfc grid: start with 1013.25mb everywhere,
            # then refine using any previously written psfc analysis.
            # This bootstraps correctly: first cycle uses 1013.25, subsequent
            # cycles refine from the previous Barnes analysis.
            _psfc_bg = np.full_like(grid_t2m, 1013.25, dtype=np.float32)
            if MESONET_DPSFC_PATH.exists():
                try:
                    _dp = np.frombuffer(MESONET_DPSFC_PATH.read_bytes(),
                                        dtype=np.float32).reshape(ny_small, nx_small)
                    _psfc_bg = np.clip(1013.25 + _dp, 850.0, 1050.0).astype(np.float32)
                except Exception:
                    pass

            delta_t, delta_td, delta_psfc = await loop.run_in_executor(
                _thread_pool, compute_correction,
                thinned, grid_lats, grid_lons, grid_t2m, grid_td2m, _psfc_bg,
            )

            # Write correction files atomically — run_cycle() may read at any time
            for fpath, arr in [
                (MESONET_DT_PATH,    delta_t),
                (MESONET_DTD_PATH,   delta_td),
                (MESONET_DPSFC_PATH, delta_psfc),
            ]:
                tmp = fpath.parent / (fpath.name + '.tmp')
                arr.tofile(str(tmp))
                tmp.replace(fpath)

            meta_corr = {
                'nx':         nx_small,
                'ny':         ny_small,
                'n_stations': len(thinned),
                'updated':    datetime.now(timezone.utc).isoformat(),
            }
            tmp_m = MESONET_META_PATH.parent / (MESONET_META_PATH.name + '.tmp')
            tmp_m.write_text(json.dumps(meta_corr))
            tmp_m.replace(MESONET_META_PATH)
            log.info(f'[mesonet] correction written: {len(thinned)} stations, '
                     f'{ny_small}×{nx_small} grid')

            # Release large arrays immediately — KDTree + distance matrix
            # can hold hundreds of MB if the GC doesn't collect promptly.
            del stations, stations_domain, thinned
            del grid_lats, grid_lons, grid_t2m, grid_td2m
            del delta_t, delta_td, delta_psfc
            gc.collect()
            log_memory('[mesonet] after GC')

        except Exception as e:
            log.error(f'[mesonet] worker error: {e}', exc_info=True)
            gc.collect()

        await asyncio.sleep(600)   # 10-minute interval


# ── VAD wind profile background worker ────────────────────────────────────────

async def vad_worker():
    """
    Fetch NEXRAD VAD wind profiles every 15 minutes, interpolate the observed
    low-level hodograph to 500m and 1000m AGL, grid those winds onto the blend
    output grid via Barnes OA, and write binary files for run_cycle().

    Storm motion is NOT computed here — VAD profiles are usually too shallow for
    a reliable 0-6km Bunkers estimate, so blend.py pairs these observed winds
    with the model's full-depth USTM/VSTM. Runs independently of run_cycle().
    """
    from scipy.spatial import KDTree

    KAPPA = 1.0    # deg² Barnes Gaussian width (~100 km influence)
    R     = 3.0    # deg hard cutoff radius

    def _barnes(st_lats, st_lons, st_val, gp_lats, gp_lons, gp_pts, tree):
        """Barnes-grid one scalar field; returns (values, count) flat arrays."""
        n_gp = len(gp_lats)
        out  = np.zeros(n_gp, dtype=np.float32)
        cnt  = np.zeros(n_gp, dtype=np.int32)
        CHUNK = 10_000
        for start in range(0, n_gp, CHUNK):
            end  = min(start + CHUNK, n_gp)
            nbrs = tree.query_ball_point(gp_pts[start:end], r=R)
            for local_i, nn_idx in enumerate(nbrs):
                gi = start + local_i
                if not nn_idx:
                    continue
                nn = np.array(nn_idx, dtype=np.int32)
                d2 = (gp_lats[gi] - st_lats[nn])**2 + (gp_lons[gi] - st_lons[nn])**2
                w  = np.exp(-d2 / KAPPA)
                ws = w.sum()
                if ws < 1e-10:
                    continue
                out[gi] = float((w * st_val[nn]).sum() / ws)
                cnt[gi] = len(nn)
        return out, cnt

    while True:
        try:
            meta_path = OUT_DIR / 'meta.json'
            if not meta_path.exists():
                await asyncio.sleep(60)
                continue
            meta     = json.loads(meta_path.read_text())
            ny_small = meta['ny']; nx_small = meta['nx']

            lats_path = OUT_DIR / 'rtma_lats_small.bin'
            lons_path = OUT_DIR / 'rtma_lons_small.bin'
            if not (lats_path.exists() and lons_path.exists()):
                await asyncio.sleep(60)
                continue
            grid_lats = np.frombuffer(lats_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)
            grid_lons = np.frombuffer(lons_path.read_bytes(),
                                      dtype=np.float32).reshape(ny_small, nx_small)

            profiles = await fetch_vad_profiles()
            if len(profiles) < 10:
                log.warning(f'[vad] only {len(profiles)} profiles — skipping cycle')
                await asyncio.sleep(900)
                continue
            profiles = compute_site_layer_winds(profiles)

            # Sites valid at each height (profile reached that level)
            s5  = [p for p in profiles if p['u500']  is not None]
            s10 = [p for p in profiles if p['u1000'] is not None]
            if len(s10) < 5:
                log.warning(f'[vad] only {len(s10)} sites reach 1000m — skipping')
                await asyncio.sleep(900)
                continue
            log.info(f'[vad] {len(s5)} sites @500m, {len(s10)} sites @1000m')

            gp_lats = grid_lats.ravel().astype(np.float64)
            gp_lons = grid_lons.ravel().astype(np.float64)
            gp_pts  = np.column_stack([gp_lats, gp_lons])

            def grid_pair(sites, ukey, vkey):
                st_lats = np.array([p['lat']  for p in sites], dtype=np.float64)
                st_lons = np.array([p['lon']  for p in sites], dtype=np.float64)
                st_u    = np.array([p[ukey]   for p in sites], dtype=np.float32)
                st_v    = np.array([p[vkey]   for p in sites], dtype=np.float32)
                tree    = KDTree(np.column_stack([st_lats, st_lons]))
                u, cu = _barnes(st_lats, st_lons, st_u, gp_lats, gp_lons, gp_pts, tree)
                v, _  = _barnes(st_lats, st_lons, st_v, gp_lats, gp_lons, gp_pts, tree)
                return u, v, cu

            loop = asyncio.get_running_loop()
            (u500, v500, _c5), (u1000, v1000, c10) = await loop.run_in_executor(
                _thread_pool,
                lambda: (grid_pair(s5, 'u500', 'v500'),
                         grid_pair(s10, 'u1000', 'v1000')),
            )

            # Coverage: ≥2 contributing sites at the 1000m (binding) level.
            cov = (c10 >= 2).astype(np.float32)
            shp = (ny_small, nx_small)
            outs = {
                VAD_U500_PATH:  np.where(cov > 0, u500,  0.0).reshape(shp).astype(np.float32),
                VAD_V500_PATH:  np.where(cov > 0, v500,  0.0).reshape(shp).astype(np.float32),
                VAD_U1000_PATH: np.where(cov > 0, u1000, 0.0).reshape(shp).astype(np.float32),
                VAD_V1000_PATH: np.where(cov > 0, v1000, 0.0).reshape(shp).astype(np.float32),
                VAD_COV_PATH:   cov.reshape(shp),
            }
            for fpath, arr in outs.items():
                tmp = fpath.parent / (fpath.name + '.tmp')
                arr.tofile(str(tmp))
                tmp.replace(fpath)

            vad_meta = {
                'nx': nx_small, 'ny': ny_small,
                'n_sites_500': len(s5), 'n_sites_1000': len(s10),
                'updated': datetime.now(timezone.utc).isoformat(),
            }
            tmp_m = VAD_META_PATH.parent / (VAD_META_PATH.name + '.tmp')
            tmp_m.write_text(json.dumps(vad_meta))
            tmp_m.replace(VAD_META_PATH)
            log.info(f'[vad] grid written: coverage={int((cov>0).sum())}/{ny_small*nx_small} pts '
                     f'({len(s10)} sites @1000m)')

            del profiles, grid_lats, grid_lons, gp_lats, gp_lons, gp_pts
            gc.collect()

        except Exception as e:
            log.error(f'[vad_worker] error: {e}', exc_info=True)
            gc.collect()

        await asyncio.sleep(900)   # 15-minute interval


# ── Hourly blend cycle ────────────────────────────────────────────────────────

async def run_cycle():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    log.info(f'Starting cycle for {now.strftime("%Y-%m-%d %H:00Z")}')
    log_memory('cycle start')
    if _cycle_lock.locked():
        log.warning('Previous cycle still running — skipping')
        return
    async with _cycle_lock:
        try:
            # Run RRFS in parallel with RTMA/TPW/HRRR — no added latency when
            # RRFS succeeds. RAP fallback is only awaited when RRFS returns None.
            rtma_task  = asyncio.create_task(fetch_rtma(now))
            rrfs_task  = asyncio.create_task(fetch_rrfs(now))
            tpw_task   = asyncio.create_task(fetch_latest_tpw(now))
            hrrr_task  = fetch_hrrr_hlcy(now)
            rtma, rrfs_result, tpw_data, hrrr_hlcy = await asyncio.gather(
                rtma_task, rrfs_task, tpw_task, hrrr_task
            )
            if hrrr_hlcy is not None:
                log.info(f'[hrrr] hlcy available: shape={hrrr_hlcy["hlcy"].shape} '
                         f'max={float(hrrr_hlcy["hlcy"].max()):.0f}')
            else:
                log.info('[hrrr] hlcy not available this cycle — using RRFS+RTMA only')

            if rrfs_result is not None:
                upper_air = rrfs_result
                log.info('[pipeline] using RRFS upper-air fields')
            else:
                log.warning('[pipeline] RRFS unavailable — falling back to RAP')
                upper_air = await fetch_rap(now)

            if upper_air is None:
                log.error('[pipeline] both RRFS and RAP failed — skipping cycle')
                return

            if rtma is None:
                log.warning('RTMA fetch returned None — skipping cycle')
                return

            FACTOR = 0.5
            loop   = asyncio.get_running_loop()

            # Save downsampled RTMA T/Td + lats/lons as the reference for
            # mesonet_worker(). Using real projected lats/lons (not linspace)
            # ensures KDTree nearest-gridpoint lookup uses the correct geometry.
            # lons converted to -180..180 to match station coordinates.
            def _save_rtma_ref(rtma_full):
                from writer import CLIP_LAT_MIN, CLIP_LAT_MAX, CLIP_LON_MIN, CLIP_LON_MAX
                lons_raw = rtma_full['lons']
                lons_180 = np.where(lons_raw > 180.0, lons_raw - 360.0, lons_raw)
                lats = rtma_full['lats']
                # Clip to GOES bbox using centre column/row (same logic as writer.py)
                # so mesonet_worker() reference grids match the blend output domain.
                row_lats = lats[:, lats.shape[1] // 2]
                col_lons = lons_180[lons_180.shape[0] // 2, :]
                row_mask = (row_lats >= CLIP_LAT_MIN) & (row_lats <= CLIP_LAT_MAX)
                col_mask = (col_lons >= CLIP_LON_MIN) & (col_lons <= CLIP_LON_MAX)
                saves = [
                    (OUT_DIR / 'rtma_t2m_small.bin',  ndimage_zoom(rtma_full['t2m'][np.ix_(row_mask, col_mask)],  FACTOR, order=1)),
                    (OUT_DIR / 'rtma_td2m_small.bin', ndimage_zoom(rtma_full['td2m'][np.ix_(row_mask, col_mask)], FACTOR, order=1)),
                    (OUT_DIR / 'rtma_lats_small.bin', ndimage_zoom(lats[np.ix_(row_mask, col_mask)],              FACTOR, order=1)),
                    (OUT_DIR / 'rtma_lons_small.bin', ndimage_zoom(lons_180[np.ix_(row_mask, col_mask)],          FACTOR, order=1)),
                ]
                for fpath, arr in saves:
                    tmp = fpath.parent / (fpath.name + '.tmp')
                    arr.astype(np.float32).tofile(str(tmp))
                    tmp.replace(fpath)
                log.info('[mesonet] saved clipped RTMA reference grids (t2m, td2m, lats, lons)')

            await loop.run_in_executor(_thread_pool, _save_rtma_ref, rtma)

            # Apply the most recent mesonet correction if available and fresh.
            # This is fast (<1 s): read two small .bin files + two zoom calls.
            # The correction was computed by mesonet_worker() in the background.
            if (MESONET_META_PATH.exists()
                    and MESONET_DT_PATH.exists()
                    and MESONET_DTD_PATH.exists()):
                try:
                    corr_meta = json.loads(MESONET_META_PATH.read_text())
                    updated   = datetime.fromisoformat(corr_meta['updated'])
                    age_s     = (datetime.now(timezone.utc) - updated).total_seconds()
                    if age_s < MESONET_MAX_AGE_S:
                        ny_c = corr_meta['ny'];  nx_c = corr_meta['nx']
                        dt   = np.frombuffer(MESONET_DT_PATH.read_bytes(),
                                             dtype=np.float32).reshape(ny_c, nx_c)
                        dtd  = np.frombuffer(MESONET_DTD_PATH.read_bytes(),
                                             dtype=np.float32).reshape(ny_c, nx_c)
                        # Guard against stale correction from a previous RTMA domain
                        # size (e.g. after a deploy that changes DOWNSAMPLE_FACTOR).
                        # The correction is at FACTOR=0.5 resolution; upsampled back
                        # to full RTMA must yield exactly ny_full×nx_full after crop.
                        # If the correction's compressed dimensions don't match the
                        # current RTMA shape when upsampled, discard and skip.
                        UP = 1.0 / FACTOR
                        ny_full, nx_full = rtma['t2m'].shape
                        ny_corr_up = round(ny_c * UP)
                        nx_corr_up = round(nx_c * UP)
                        if abs(ny_corr_up - ny_full) > 4 or abs(nx_corr_up - nx_full) > 4:
                            log.warning(
                                f'[mesonet] correction shape mismatch: '
                                f'correction={ny_c}×{nx_c} → upsampled={ny_corr_up}×{nx_corr_up} '
                                f'vs RTMA={ny_full}×{nx_full} — discarding stale correction'
                            )
                            raise ValueError('shape mismatch')
                        # Upsample correction to full RTMA resolution, crop ±1px
                        dt_full  = ndimage_zoom(dt,  UP, order=1)[:ny_full, :nx_full]
                        dtd_full = ndimage_zoom(dtd, UP, order=1)[:ny_full, :nx_full]
                        rtma = dict(rtma)
                        rtma['t2m']  = np.clip(
                            rtma['t2m']  + dt_full,  200.0, 340.0
                        ).astype(np.float32)
                        rtma['td2m'] = np.clip(
                            rtma['td2m'] + dtd_full, 200.0, 320.0
                        ).astype(np.float32)

                        # Apply surface pressure analysis if available.
                        # Build absolute psfc = 1013.25mb background + Barnes ΔP.
                        if MESONET_DPSFC_PATH.exists():
                            try:
                                dp_raw = np.frombuffer(
                                    MESONET_DPSFC_PATH.read_bytes(),
                                    dtype=np.float32).reshape(ny_c, nx_c)
                                dp_full = ndimage_zoom(dp_raw, UP, order=1)[:ny_full, :nx_full]
                                rtma['psfc'] = np.clip(
                                    1013.25 + dp_full, 850.0, 1050.0
                                ).astype(np.float32)
                            except Exception as _e:
                                log.warning(f'[mesonet] psfc correction failed: {_e}')
                                rtma['psfc'] = np.full_like(rtma['t2m'], 1000.0)
                        else:
                            # No psfc analysis yet — use 1000mb (same as before)
                            rtma['psfc'] = np.full_like(rtma['t2m'], 1000.0)

                        log.info(f'[mesonet] applied correction '
                                 f'(age={age_s/60:.0f} min, '
                                 f'stations={corr_meta["n_stations"]})')
                    else:
                        log.warning(f'[mesonet] correction stale '
                                    f'({age_s/60:.0f} min) — skipping')
                except Exception as e:
                    log.warning(f'[mesonet] correction read failed: {e} '
                                f'— using raw RTMA')
            # Ensure psfc is always present in rtma dict (stale/missing/failed correction)
            if 'psfc' not in rtma:
                rtma['psfc'] = np.full_like(rtma['t2m'], 1000.0)

            # Load VAD observed-hodograph grids if fresh; upsample to full RTMA.
            # blend() uses these (where coverage>0) with model storm motion to
            # form an observation-anchored 0-1km SRH. Absent/stale → vad_data
            # stays None and blend() falls back to the model SRH blend.
            vad_data = None
            _vad_files = [VAD_U500_PATH, VAD_V500_PATH, VAD_U1000_PATH,
                          VAD_V1000_PATH, VAD_COV_PATH, VAD_META_PATH]
            if all(p.exists() for p in _vad_files):
                try:
                    vad_meta_j  = json.loads(VAD_META_PATH.read_text())
                    vad_updated = datetime.fromisoformat(vad_meta_j['updated'])
                    vad_age_s   = (datetime.now(timezone.utc) - vad_updated).total_seconds()
                    if vad_age_s < VAD_MAX_AGE_S:
                        ny_v = vad_meta_j['ny']; nx_v = vad_meta_j['nx']
                        UP = 1.0 / FACTOR
                        ny_full, nx_full = rtma['t2m'].shape
                        def _load_up(path, order):
                            a = np.frombuffer(path.read_bytes(),
                                              dtype=np.float32).reshape(ny_v, nx_v)
                            return ndimage_zoom(a, UP, order=order)[:ny_full, :nx_full].astype(np.float32)
                        vad_data = {
                            'u500':  _load_up(VAD_U500_PATH,  1),
                            'v500':  _load_up(VAD_V500_PATH,  1),
                            'u1000': _load_up(VAD_U1000_PATH, 1),
                            'v1000': _load_up(VAD_V1000_PATH, 1),
                            # nearest-neighbour for the coverage mask (keep it crisp 0/1)
                            'cov':   _load_up(VAD_COV_PATH,   0),
                        }
                        log.info(f'[vad] loaded: coverage={float((vad_data["cov"]>0).mean())*100:.0f}% '
                                 f'age={vad_age_s/60:.0f}min sites={vad_meta_j.get("n_sites_1000","?")}')
                    else:
                        log.warning(f'[vad] grids stale ({vad_age_s/60:.0f} min) — skipping')
                except Exception as _e:
                    log.warning(f'[vad] load failed: {_e}')

            # Blend derivation is CPU-heavy; off-loaded to thread pool.
            # tpw_data is passed as third positional arg; blend() defaults to
            # None if not available so this is safe when TPW fetch failed.
            blended = await loop.run_in_executor(
                _thread_pool, do_blend, rtma, upper_air, tpw_data, hrrr_hlcy, vad_data
            )
            write_output(blended, now)

            # Explicit cleanup — cfgrib/xarray leave internal references that
            # prevent the GC from collecting large numpy arrays promptly.
            del rtma, upper_air, tpw_data, blended
            try:
                import cfgrib.messages
                cfgrib.messages.EMPTY_HEADER_ERRORS.clear()
            except Exception:
                pass
            gc.collect()

            log.info('Cycle complete')
            log_memory('cycle end')
        except Exception as e:
            log.error(f'Cycle failed: {e}', exc_info=True)


# ── Scheduler ─────────────────────────────────────────────────────────────────

async def scheduler():
    await run_cycle()   # run immediately on startup
    while True:
        now      = datetime.now(timezone.utc)
        next_run = now.replace(minute=28, second=0, microsecond=0)
        if now >= next_run:
            # timedelta addition handles hour=23 → next day correctly.
            next_run = (next_run + timedelta(hours=1)).replace(
                minute=28, second=0, microsecond=0
            )
        wait = (next_run - now).total_seconds()
        log.info(f'Next cycle in {wait:.0f}s at {next_run.strftime("%H:%M")}Z')
        await asyncio.sleep(wait)
        await run_cycle()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    from server import start_server
    runner = await start_server()
    try:
        # One-time terrain setup for BACI — runs in background, non-blocking
        if not terrain_already_exists():
            log.info('[terrain] Starting one-time BACI terrain build...')
            asyncio.create_task(build_terrain())
        else:
            log.info('[terrain] BACI terrain already built — skipping')
        await asyncio.gather(
            scheduler(),
            mesonet_worker(),
            vad_worker(),
            satellite_worker(),
        )
    finally:
        _thread_pool.shutdown(wait=False)
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
