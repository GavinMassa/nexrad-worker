import asyncio, logging, json, gc, os
import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from scipy.ndimage import zoom as ndimage_zoom
from fetch_rtma import fetch_rtma
from fetch_rap import fetch_rap
from fetch_tpw import fetch_latest_tpw
from mesonet import fetch_mesonet_obs, compute_correction
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
MESONET_DT_PATH   = OUT_DIR / 'mesonet_delta_t.bin'
MESONET_DTD_PATH  = OUT_DIR / 'mesonet_delta_td.bin'
MESONET_META_PATH = OUT_DIR / 'mesonet_meta.json'
MESONET_MAX_AGE_S = 90 * 60   # discard corrections older than 90 minutes


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

            # IDW is CPU-bound but fast with the thinned network (~600 stations)
            loop = asyncio.get_running_loop()
            delta_t, delta_td = await loop.run_in_executor(
                _thread_pool, compute_correction,
                thinned, grid_lats, grid_lons, grid_t2m, grid_td2m,
            )

            # Write correction files atomically — run_cycle() may read at any time
            for fpath, arr in [(MESONET_DT_PATH, delta_t), (MESONET_DTD_PATH, delta_td)]:
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
            del delta_t, delta_td
            gc.collect()
            log_memory('[mesonet] after GC')

        except Exception as e:
            log.error(f'[mesonet] worker error: {e}', exc_info=True)
            gc.collect()

        await asyncio.sleep(600)   # 10-minute interval


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
            rtma_task = asyncio.create_task(fetch_rtma(now))
            rap_task  = asyncio.create_task(fetch_rap(now))
            tpw_task  = asyncio.create_task(fetch_latest_tpw(now))
            rtma, rap, tpw_data = await asyncio.gather(rtma_task, rap_task, tpw_task)

            if rtma is None:
                log.warning('RTMA fetch returned None — skipping cycle')
                return
            if rap is None:
                log.warning('RAP fetch returned None — writing RTMA-only output')
                write_output(rtma, now)
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
                        # Upsample correction to full RTMA resolution, crop ±1px
                        UP = 1.0 / FACTOR
                        ny_full, nx_full = rtma['t2m'].shape
                        dt_full  = ndimage_zoom(dt,  UP, order=1)[:ny_full, :nx_full]
                        dtd_full = ndimage_zoom(dtd, UP, order=1)[:ny_full, :nx_full]
                        rtma = dict(rtma)
                        rtma['t2m']  = np.clip(
                            rtma['t2m']  + dt_full,  200.0, 340.0
                        ).astype(np.float32)
                        rtma['td2m'] = np.clip(
                            rtma['td2m'] + dtd_full, 200.0, 320.0
                        ).astype(np.float32)
                        log.info(f'[mesonet] applied correction '
                                 f'(age={age_s/60:.0f} min, '
                                 f'stations={corr_meta["n_stations"]})')
                    else:
                        log.warning(f'[mesonet] correction stale '
                                    f'({age_s/60:.0f} min) — skipping')
                except Exception as e:
                    log.warning(f'[mesonet] correction read failed: {e} '
                                f'— using raw RTMA')

            # Blend derivation is CPU-heavy; off-loaded to thread pool.
            # tpw_data is passed as third positional arg; blend() defaults to
            # None if not available so this is safe when TPW fetch failed.
            blended = await loop.run_in_executor(
                _thread_pool, do_blend, rtma, rap, tpw_data
            )
            write_output(blended, now)

            # Explicit cleanup — cfgrib/xarray leave internal references that
            # prevent the GC from collecting large numpy arrays promptly.
            del rtma, rap, tpw_data, blended
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
            satellite_worker(),
        )
    finally:
        _thread_pool.shutdown(wait=False)
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
